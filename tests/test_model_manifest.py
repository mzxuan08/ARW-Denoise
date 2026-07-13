from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arw_denoise.model_manifest import ModelManifestError, default_model_root, load_manifest


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(root: Path, **changes: object) -> Path:
    artifact = root / "pmrid-fp16.onnx"
    artifact.write_bytes(b"onnx model bytes")
    data: dict[str, object] = {
        "schema_version": 1,
        "model_id": "pmrid-general-raw",
        "display_name": "PMRID 通用 RAW GPU",
        "version": "1.0.0",
        "artifact": {
            "file": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "format": "onnx",
            "precision": "fp16",
        },
        "source": {
            "repository": "https://github.com/MegEngine/PMRID",
            "commit": "8ebb9e8e96559881dee957f34243933c5beb77dd",
            "checkpoint_sha256": "9361614f3514d27351d81909f2215c0fdc38619c0288d936b7266485ac106c14",
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
    data.update(changes)
    path = root / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_manifest_verifies_contract_and_artifact(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path))

    assert manifest.model_id == "pmrid-general-raw"
    assert manifest.artifact_path.name == "pmrid-fp16.onnx"
    assert manifest.input.channels == 4
    assert manifest.noise_input.name == "effective_iso"
    assert manifest.tiling.recommended_size == 1024


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("artifact", "precision", "int8"),
        ("input", "channels", 3),
        ("input", "layout", "NHWC"),
        ("input", "cfa_order", ["B", "G1", "G2", "R"]),
    ],
)
def test_manifest_rejects_unsupported_tensor_contract(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    path = _write_manifest(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data[section][key] = value
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelManifestError):
        load_manifest(path)


def test_manifest_rejects_missing_and_unknown_fields(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["license"]
    data["surprise"] = True
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelManifestError):
        load_manifest(path)


def test_manifest_rejects_invalid_noise_input(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["noise_input"]["shape"] = [1]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelManifestError, match="noise_input"):
        load_manifest(path)


def test_manifest_rejects_bad_hash(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["artifact"]["sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelManifestError, match="SHA-256"):
        load_manifest(path)


def test_manifest_rejects_artifact_path_escape(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    path = _write_manifest(model_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["artifact"]["file"] = "../outside.onnx"
    data["artifact"]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ModelManifestError, match="模型目录"):
        load_manifest(path)


def test_default_model_root_prefers_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARW_DENOISE_MODEL_DIR", str(tmp_path))
    assert default_model_root() == tmp_path.resolve()


def test_bundled_manifest_schema_is_valid_without_loading_binary() -> None:
    manifest = load_manifest(ROOT / "models" / "pmrid" / "manifest.json", verify_artifact=False)
    assert manifest.model_id == "pmrid-general-raw"
    assert manifest.source.commit == "8ebb9e8e96559881dee957f34243933c5beb77dd"
    assert manifest.artifact.sha256 == "34bf0b58b31566eee3ca8b3f99f3a6e000188ca6f510081d0de8355cc5a6cff0"
