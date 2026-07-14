from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_bundle import bundle_report, prune_bundle


def test_audit_groups_sizes_and_prunes_only_allowlisted_assets(tmp_path: Path) -> None:
    (tmp_path / "ArwDenoise.exe").write_bytes(b"exe")
    tensor = tmp_path / "_internal" / "onnxruntime" / "capi" / "onnxruntime_providers_tensorrt.dll"
    tensor.parent.mkdir(parents=True)
    tensor.write_bytes(b"unused")
    required = tensor.parent / "onnxruntime_providers_cuda.dll"
    required.write_bytes(b"required")
    report = bundle_report(tmp_path)
    assert report["total_bytes"] == len(b"exeunusedrequired")
    removed = prune_bundle(tmp_path)
    assert "_internal/onnxruntime/capi/onnxruntime_providers_tensorrt.dll" in removed
    assert required.read_bytes() == b"required"


def test_prune_refuses_arbitrary_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Refusing"):
        prune_bundle(tmp_path)
