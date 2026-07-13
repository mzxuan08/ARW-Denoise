from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arw_denoise.engines import DenoiseRequest
from arw_denoise.model_manifest import (
    ArtifactSpec,
    ModelManifest,
    NoiseInputSpec,
    RuntimeSpec,
    SourceSpec,
    TensorSpec,
    TilingSpec,
)
from arw_denoise.onnx_engine import GpuRuntimeError, OnnxRuntimeEngine


def _manifest(tmp_path: Path) -> ModelManifest:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    tensor = TensorSpec(
        name="raw",
        layout="NCHW",
        channels=4,
        dtype="float32",
        value_range=(0.0, 1.0),
        cfa_order=("R", "G1", "G2", "B"),
    )
    return ModelManifest(
        schema_version=1,
        model_id="test-model",
        display_name="Test model",
        version="1.0.0",
        artifact=ArtifactSpec(file="model.onnx", sha256="0" * 64, format="onnx", precision="fp16"),
        source=SourceSpec(repository="https://example.test", commit="abc", checkpoint_sha256="1" * 64),
        license="Apache-2.0",
        input=tensor,
        output=TensorSpec(
            name="denoised",
            layout="NCHW",
            channels=4,
            dtype="float32",
            value_range=(0.0, 1.0),
            cfa_order=("R", "G1", "G2", "B"),
        ),
        noise_input=NoiseInputSpec(
            name="effective_iso",
            dtype="float32",
            shape=("N", 1, 1, 1),
            value_range=(100.0, 25600.0),
        ),
        runtime=RuntimeSpec(
            minimum_onnxruntime="1.23.2",
            providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        ),
        tiling=TilingSpec(recommended_size=1024, overlap=64, minimum_size=256),
        artifact_path=artifact,
    )


class FakeSession:
    def __init__(self, providers: list[object], bad_output: str | None = None):
        self.providers = providers
        self.bad_output = bad_output
        self.last_inputs: dict[str, np.ndarray] | None = None

    def get_providers(self) -> list[str]:
        return [item[0] if isinstance(item, tuple) else item for item in self.providers]

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="raw", type="tensor(float)", shape=["batch", 4, "height", "width"]),
            SimpleNamespace(name="effective_iso", type="tensor(float)", shape=["batch", 1, 1, 1]),
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="denoised", type="tensor(float)", shape=["batch", 4, "height", "width"])]

    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.last_inputs = inputs
        result = inputs["raw"] * np.float32(0.5)
        if self.bad_output == "fallback":
            self.providers = ["CPUExecutionProvider"]
        if self.bad_output == "shape":
            result = result[:, :3]
        if self.bad_output == "nan":
            result = result.copy()
            result.flat[0] = np.nan
        return [result]


class FakeRuntime:
    __version__ = "1.23.2"

    def __init__(self, providers: list[str], bad_output: str | None = None, preload_error: Exception | None = None):
        self.providers = providers
        self.bad_output = bad_output
        self.preload_error = preload_error
        self.preloaded: str | None = None
        self.session: FakeSession | None = None

    def preload_dlls(self, **kwargs: object) -> None:
        if self.preload_error:
            raise self.preload_error
        self.preloaded = str(kwargs.get("directory"))

    def get_available_providers(self) -> list[str]:
        return self.providers

    def InferenceSession(self, path: str, providers: list[object]) -> FakeSession:  # noqa: N802
        self.session = FakeSession(providers, self.bad_output)
        return self.session


def test_cuda_engine_transposes_inputs_and_reports_actual_provider(tmp_path: Path) -> None:
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
    engine = OnnxRuntimeEngine(_manifest(tmp_path), runtime_module=runtime, dll_directory=tmp_path)
    packed = np.full((32, 48, 4), 0.8, dtype=np.float32)

    result = engine.run(DenoiseRequest(packed=packed, effective_iso=3200.0))

    assert result.packed.shape == packed.shape
    assert np.allclose(result.packed, 0.4)
    assert result.engine.provider == "CUDAExecutionProvider"
    assert result.engine.is_gpu
    assert runtime.preloaded == str(tmp_path)
    assert runtime.session is not None
    assert runtime.session.last_inputs is not None
    assert runtime.session.last_inputs["raw"].shape == (1, 4, 32, 48)
    assert runtime.session.last_inputs["effective_iso"].shape == (1, 1, 1, 1)


def test_cuda_engine_rejects_missing_provider(tmp_path: Path) -> None:
    with pytest.raises(GpuRuntimeError, match="CUDAExecutionProvider"):
        OnnxRuntimeEngine(_manifest(tmp_path), runtime_module=FakeRuntime(["CPUExecutionProvider"]))


def test_cuda_engine_reports_dll_preload_failure(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_error=OSError("missing cudnn64_9.dll"),
    )
    with pytest.raises(GpuRuntimeError, match="cuDNN"):
        OnnxRuntimeEngine(_manifest(tmp_path), runtime_module=runtime, dll_directory=tmp_path)


@pytest.mark.parametrize("bad_output", ["shape", "nan"])
def test_cuda_engine_rejects_invalid_model_output(tmp_path: Path, bad_output: str) -> None:
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"], bad_output=bad_output)
    engine = OnnxRuntimeEngine(_manifest(tmp_path), runtime_module=runtime)
    with pytest.raises(GpuRuntimeError, match="输出"):
        engine.run(
            DenoiseRequest(
                packed=np.zeros((32, 32, 4), dtype=np.float32),
                effective_iso=1600.0,
            )
        )


def test_cuda_engine_requires_dimensions_divisible_by_sixteen(tmp_path: Path) -> None:
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])
    engine = OnnxRuntimeEngine(_manifest(tmp_path), runtime_module=runtime)
    with pytest.raises(GpuRuntimeError, match="16"):
        engine.run(
            DenoiseRequest(
                packed=np.zeros((31, 32, 4), dtype=np.float32),
                effective_iso=1600.0,
            )
        )


def test_cuda_engine_rejects_runtime_fallback_after_inference(tmp_path: Path) -> None:
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"], bad_output="fallback")
    engine = OnnxRuntimeEngine(_manifest(tmp_path), runtime_module=runtime)
    with pytest.raises(GpuRuntimeError, match="回退"):
        engine.run(
            DenoiseRequest(
                packed=np.zeros((32, 32, 4), dtype=np.float32),
                effective_iso=1600.0,
            )
        )
