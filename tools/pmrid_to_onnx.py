from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxconverter_common import float16
from torch import nn

try:
    from tools.pmrid_net import Network
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from pmrid_net import Network


UPSTREAM_COMMIT = "8ebb9e8e96559881dee957f34243933c5beb77dd"
UPSTREAM_CHECKPOINT_SHA256 = "9361614f3514d27351d81909f2215c0fdc38619c0288d936b7266485ac106c14"


@dataclass(frozen=True)
class ExportResult:
    output: Path
    output_sha256: str
    fp32_max_error: float
    fp16_max_error: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str = UPSTREAM_CHECKPOINT_SHA256) -> str:
    if not path.is_file():
        raise RuntimeError(f"PMRID checkpoint not found: {path}")
    actual = file_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(f"PMRID checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


class PmridInferenceModel(nn.Module):
    """Official PMRID network plus its ISO-domain pre/post transform."""

    K_SLOPE = 0.0005995267
    K_INTERCEPT = 0.00868861
    SIGMA_QUADRATIC = 7.11772e-7
    SIGMA_LINEAR = 6.514934e-4
    SIGMA_INTERCEPT = 0.11492713
    ANCHOR_ISO = 1600.0
    SENSOR_SCALE = 959.0
    NETWORK_SCALE = 256.0

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    @classmethod
    def _noise_parameters(cls, iso: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = cls.K_SLOPE * iso + cls.K_INTERCEPT
        sigma = cls.SIGMA_QUADRATIC * iso * iso + cls.SIGMA_LINEAR * iso + cls.SIGMA_INTERCEPT
        anchor = torch.full_like(iso, cls.ANCHOR_ISO)
        anchor_k = cls.K_SLOPE * anchor + cls.K_INTERCEPT
        anchor_sigma = (
            cls.SIGMA_QUADRATIC * anchor * anchor
            + cls.SIGMA_LINEAR * anchor
            + cls.SIGMA_INTERCEPT
        )
        conversion_k = anchor_k / k
        conversion_b = (sigma / (k * k) - anchor_sigma / (anchor_k * anchor_k)) * anchor_k
        return conversion_k, conversion_b

    def forward(self, raw: torch.Tensor, effective_iso: torch.Tensor) -> torch.Tensor:
        iso = torch.clamp(effective_iso, 400.0, 25600.0)
        conversion_k, conversion_b = self._noise_parameters(iso)
        normalized = (
            (raw * self.SENSOR_SCALE * conversion_k + conversion_b)
            / self.SENSOR_SCALE
            * self.NETWORK_SCALE
        )
        prediction = self.network(normalized) / self.NETWORK_SCALE
        restored = (
            (prediction * self.SENSOR_SCALE - conversion_b)
            / conversion_k
            / self.SENSOR_SCALE
        )
        return torch.clamp(restored, 0.0, 1.0)


def _load_wrapper(checkpoint: Path, expected_sha256: str) -> PmridInferenceModel:
    verify_checkpoint(checkpoint, expected_sha256)
    network = Network()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError("PMRID checkpoint does not contain a state dictionary")
    network.load_state_dict(state, strict=True)
    return PmridInferenceModel(network.eval()).eval()


def _ort_prediction(path: Path, raw: np.ndarray, iso: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(["denoised"], {"raw": raw, "effective_iso": iso})[0]


def export_models(
    *,
    checkpoint: Path,
    output: Path,
    expected_checkpoint_sha256: str = UPSTREAM_CHECKPOINT_SHA256,
    sample_height: int = 64,
    sample_width: int = 64,
) -> ExportResult:
    if sample_height % 16 or sample_width % 16:
        raise ValueError("PMRID export sample dimensions must be multiples of 16")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _load_wrapper(checkpoint.resolve(), expected_checkpoint_sha256)

    generator = torch.Generator(device="cpu").manual_seed(20260714)
    raw = torch.rand((3, 4, sample_height, sample_width), generator=generator, dtype=torch.float32)
    iso = torch.tensor([[[[400.0]]], [[[1600.0]]], [[[25600.0]]]], dtype=torch.float32)
    with torch.inference_mode():
        reference = wrapper(raw, iso).numpy()

    token = uuid.uuid4().hex
    fp32_path = output.with_name(f".{output.stem}.{token}.fp32.onnx")
    fp16_path = output.with_name(f".{output.stem}.{token}.fp16.onnx")
    try:
        torch.onnx.export(
            wrapper,
            (raw, iso),
            str(fp32_path),
            input_names=["raw", "effective_iso"],
            output_names=["denoised"],
            dynamic_axes={
                "raw": {0: "batch", 2: "height", 3: "width"},
                "effective_iso": {0: "batch"},
                "denoised": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        fp32_model = onnx.load(str(fp32_path))
        onnx.checker.check_model(fp32_model)
        fp32_prediction = _ort_prediction(fp32_path, raw.numpy(), iso.numpy())
        fp32_error = float(np.max(np.abs(reference - fp32_prediction)))
        if fp32_error > 1e-5:
            raise RuntimeError(f"FP32 ONNX parity failed: max error {fp32_error}")

        fp16_model = float16.convert_float_to_float16(
            fp32_model,
            keep_io_types=True,
            max_finite_val=65504.0,
        )
        onnx.checker.check_model(fp16_model)
        onnx.save(fp16_model, str(fp16_path))
        fp16_prediction = _ort_prediction(fp16_path, raw.numpy(), iso.numpy())
        fp16_error = float(np.max(np.abs(reference - fp16_prediction)))
        if fp16_error > 2e-3:
            raise RuntimeError(f"FP16 ONNX parity failed: max error {fp16_error}")

        os.replace(fp16_path, output)
        return ExportResult(
            output=output,
            output_sha256=file_sha256(output),
            fp32_max_error=fp32_error,
            fp16_max_error=fp16_error,
        )
    finally:
        fp32_path.unlink(missing_ok=True)
        fp16_path.unlink(missing_ok=True)


def write_manifest(path: Path, result: ExportResult) -> None:
    manifest = {
        "schema_version": 1,
        "model_id": "pmrid-general-raw",
        "display_name": "PMRID 通用 RAW GPU",
        "version": "1.0.0",
        "artifact": {
            "file": result.output.name,
            "sha256": result.output_sha256,
            "format": "onnx",
            "precision": "fp16",
        },
        "source": {
            "repository": "https://github.com/MegEngine/PMRID",
            "commit": UPSTREAM_COMMIT,
            "checkpoint_sha256": UPSTREAM_CHECKPOINT_SHA256,
        },
        "license": "Apache-2.0",
        "input": {
            "name": "raw",
            "layout": "NCHW",
            "channels": 4,
            "dtype": "float32",
            "range": [0.0, 1.0],
            "cfa_order": ["R", "G1", "G2", "B"],
        },
        "output": {
            "name": "denoised",
            "layout": "NCHW",
            "channels": 4,
            "dtype": "float32",
            "range": [0.0, 1.0],
            "cfa_order": ["R", "G1", "G2", "B"],
        },
        "noise_input": {
            "name": "effective_iso",
            "dtype": "float32",
            "shape": ["N", 1, 1, 1],
            "range": [400.0, 25600.0],
        },
        "runtime": {
            "minimum_onnxruntime": "1.23.2",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        },
        "tiling": {"recommended_size": 1024, "overlap": 64, "minimum_size": 256},
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert the pinned PMRID checkpoint to a verified FP16 ONNX model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = export_models(checkpoint=args.checkpoint, output=args.output)
    manifest_path = args.manifest or args.output.with_name("manifest.json")
    write_manifest(manifest_path, result)
    print(
        json.dumps(
            {
                "output": str(result.output),
                "sha256": result.output_sha256,
                "fp32_max_error": result.fp32_max_error,
                "fp16_max_error": result.fp16_max_error,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
