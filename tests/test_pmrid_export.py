from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")

from tools.pmrid_net import Network
from tools.pmrid_to_onnx import (  # noqa: E402
    PmridInferenceModel,
    export_models,
    verify_checkpoint,
)


def test_pmrid_wrapper_is_finite_and_preserves_shape() -> None:
    torch.manual_seed(7)
    wrapper = PmridInferenceModel(Network().eval()).eval()
    raw = torch.rand(1, 4, 32, 48)
    iso = torch.tensor([[[[1600.0]]]])

    with torch.inference_mode():
        output = wrapper(raw, iso)

    assert output.shape == raw.shape
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_verify_checkpoint_rejects_wrong_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckp"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_checkpoint(checkpoint, "0" * 64)


def test_exported_fp32_and_fp16_match_pytorch(tmp_path: Path) -> None:
    torch.manual_seed(11)
    checkpoint = tmp_path / "model.ckp"
    torch.save(Network().state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    result = export_models(
        checkpoint=checkpoint,
        output=tmp_path / "pmrid-fp16.onnx",
        expected_checkpoint_sha256=digest,
        sample_height=32,
        sample_width=32,
    )

    assert result.fp32_max_error <= 1e-5
    assert result.fp16_max_error <= 2e-3
    assert result.output.is_file()
    assert len(result.output_sha256) == 64

