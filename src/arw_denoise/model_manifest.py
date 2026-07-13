from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .domain import ArwDenoiseError


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_CFA_ORDER = ("R", "G1", "G2", "B")


class ModelManifestError(ArwDenoiseError):
    """A model package is missing, damaged, or incompatible."""


@dataclass(frozen=True)
class ArtifactSpec:
    file: str
    sha256: str
    format: str
    precision: str


@dataclass(frozen=True)
class SourceSpec:
    repository: str
    commit: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class TensorSpec:
    name: str
    layout: str
    channels: int
    dtype: str
    value_range: tuple[float, float]
    cfa_order: tuple[str, str, str, str]


@dataclass(frozen=True)
class NoiseInputSpec:
    name: str
    dtype: str
    shape: tuple[str | int, int, int, int]
    value_range: tuple[float, float]


@dataclass(frozen=True)
class RuntimeSpec:
    minimum_onnxruntime: str
    providers: tuple[str, ...]


@dataclass(frozen=True)
class TilingSpec:
    recommended_size: int
    overlap: int
    minimum_size: int


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    display_name: str
    version: str
    artifact: ArtifactSpec
    source: SourceSpec
    license: str
    input: TensorSpec
    output: TensorSpec
    noise_input: NoiseInputSpec
    runtime: RuntimeSpec
    tiling: TilingSpec
    artifact_path: Path


def _strict_object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelManifestError(f"模型清单 {name} 必须是对象")
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise ModelManifestError(f"模型清单 {name} 缺少字段：{', '.join(sorted(missing))}")
    if unknown:
        raise ModelManifestError(f"模型清单 {name} 包含未知字段：{', '.join(sorted(unknown))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"模型清单 {name} 必须是非空文本")
    return value


def _sha256(value: Any, name: str) -> str:
    digest = _text(value, name).lower()
    if not _SHA256.fullmatch(digest):
        raise ModelManifestError(f"模型清单 {name} 不是有效 SHA-256")
    return digest


def _parse_tensor(value: Any, name: str) -> TensorSpec:
    data = _strict_object(value, name, {"name", "layout", "channels", "dtype", "range", "cfa_order"})
    layout = _text(data["layout"], f"{name}.layout")
    channels = data["channels"]
    dtype = _text(data["dtype"], f"{name}.dtype")
    value_range = data["range"]
    cfa_order = data["cfa_order"]
    if layout != "NCHW" or channels != 4:
        raise ModelManifestError(f"模型清单 {name} 必须是 NCHW 四通道 Bayer")
    if dtype not in {"float16", "float32"}:
        raise ModelManifestError(f"模型清单 {name}.dtype 不受支持：{dtype}")
    if not isinstance(value_range, list) or len(value_range) != 2:
        raise ModelManifestError(f"模型清单 {name}.range 必须包含上下界")
    try:
        bounds = (float(value_range[0]), float(value_range[1]))
    except (TypeError, ValueError) as exc:
        raise ModelManifestError(f"模型清单 {name}.range 无效") from exc
    if bounds != (0.0, 1.0):
        raise ModelManifestError(f"模型清单 {name}.range 必须是 [0, 1]")
    if not isinstance(cfa_order, list) or tuple(cfa_order) != _CFA_ORDER:
        raise ModelManifestError(f"模型清单 {name}.cfa_order 必须是 R/G1/G2/B")
    return TensorSpec(
        name=_text(data["name"], f"{name}.name"),
        layout=layout,
        channels=channels,
        dtype=dtype,
        value_range=bounds,
        cfa_order=_CFA_ORDER,
    )


def _parse_noise_input(value: Any) -> NoiseInputSpec:
    data = _strict_object(value, "noise_input", {"name", "dtype", "shape", "range"})
    dtype = _text(data["dtype"], "noise_input.dtype")
    shape = data["shape"]
    value_range = data["range"]
    if dtype != "float32":
        raise ModelManifestError("模型清单 noise_input.dtype 必须是 float32")
    if not isinstance(shape, list) or shape != ["N", 1, 1, 1]:
        raise ModelManifestError("模型清单 noise_input.shape 必须是 [N, 1, 1, 1]")
    if not isinstance(value_range, list) or len(value_range) != 2:
        raise ModelManifestError("模型清单 noise_input.range 必须包含上下界")
    try:
        bounds = (float(value_range[0]), float(value_range[1]))
    except (TypeError, ValueError) as exc:
        raise ModelManifestError("模型清单 noise_input.range 无效") from exc
    if bounds[0] < 1.0 or bounds[1] <= bounds[0]:
        raise ModelManifestError("模型清单 noise_input.range 无效")
    return NoiseInputSpec(
        name=_text(data["name"], "noise_input.name"),
        dtype=dtype,
        shape=("N", 1, 1, 1),
        value_range=bounds,
    )


def _artifact_path(manifest_path: Path, file_name: str) -> Path:
    if Path(file_name).is_absolute():
        raise ModelManifestError("模型文件必须位于模型目录内")
    root = manifest_path.parent.resolve()
    candidate = (root / file_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ModelManifestError("模型文件必须位于模型目录内") from exc
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelManifestError(f"无法读取模型文件：{path.name}") from exc
    return digest.hexdigest()


def load_manifest(path: Path | str, *, verify_artifact: bool = True) -> ModelManifest:
    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"无法读取模型清单：{manifest_path.name}") from exc
    data = _strict_object(
        raw,
        "root",
        {
            "schema_version",
            "model_id",
            "display_name",
            "version",
            "artifact",
            "source",
            "license",
            "input",
            "output",
            "noise_input",
            "runtime",
            "tiling",
        },
    )
    if data["schema_version"] != 1:
        raise ModelManifestError("不支持的模型清单版本")
    version = _text(data["version"], "version")
    if not _SEMVER.fullmatch(version):
        raise ModelManifestError("模型版本必须使用 x.y.z 格式")

    artifact_data = _strict_object(data["artifact"], "artifact", {"file", "sha256", "format", "precision"})
    artifact = ArtifactSpec(
        file=_text(artifact_data["file"], "artifact.file"),
        sha256=_sha256(artifact_data["sha256"], "artifact.sha256"),
        format=_text(artifact_data["format"], "artifact.format"),
        precision=_text(artifact_data["precision"], "artifact.precision"),
    )
    if artifact.format != "onnx" or artifact.precision not in {"fp16", "fp32"}:
        raise ModelManifestError("模型产物必须是 ONNX FP16 或 FP32")

    source_data = _strict_object(data["source"], "source", {"repository", "commit", "checkpoint_sha256"})
    source = SourceSpec(
        repository=_text(source_data["repository"], "source.repository"),
        commit=_text(source_data["commit"], "source.commit"),
        checkpoint_sha256=_sha256(source_data["checkpoint_sha256"], "source.checkpoint_sha256"),
    )

    runtime_data = _strict_object(data["runtime"], "runtime", {"minimum_onnxruntime", "providers"})
    providers = runtime_data["providers"]
    if not isinstance(providers, list) or not providers or any(not isinstance(item, str) for item in providers):
        raise ModelManifestError("模型清单 runtime.providers 无效")
    minimum_ort = _text(runtime_data["minimum_onnxruntime"], "runtime.minimum_onnxruntime")
    if not _SEMVER.fullmatch(minimum_ort):
        raise ModelManifestError("ONNX Runtime 版本必须使用 x.y.z 格式")

    tiling_data = _strict_object(data["tiling"], "tiling", {"recommended_size", "overlap", "minimum_size"})
    try:
        tiling = TilingSpec(
            recommended_size=int(tiling_data["recommended_size"]),
            overlap=int(tiling_data["overlap"]),
            minimum_size=int(tiling_data["minimum_size"]),
        )
    except (TypeError, ValueError) as exc:
        raise ModelManifestError("模型清单 tiling 参数无效") from exc
    if (
        tiling.minimum_size < 64
        or tiling.recommended_size < tiling.minimum_size
        or tiling.overlap < 0
        or tiling.minimum_size <= 2 * tiling.overlap
        or any(value % 16 for value in (tiling.minimum_size, tiling.recommended_size))
    ):
        raise ModelManifestError("模型清单 tiling 参数不安全")

    resolved_artifact = _artifact_path(manifest_path, artifact.file)
    if verify_artifact:
        actual_hash = _file_sha256(resolved_artifact)
        if actual_hash != artifact.sha256:
            raise ModelManifestError(
                f"模型文件 SHA-256 不匹配：期望 {artifact.sha256}，实际 {actual_hash}"
            )

    return ModelManifest(
        schema_version=1,
        model_id=_text(data["model_id"], "model_id"),
        display_name=_text(data["display_name"], "display_name"),
        version=version,
        artifact=artifact,
        source=source,
        license=_text(data["license"], "license"),
        input=_parse_tensor(data["input"], "input"),
        output=_parse_tensor(data["output"], "output"),
        noise_input=_parse_noise_input(data["noise_input"]),
        runtime=RuntimeSpec(minimum_onnxruntime=minimum_ort, providers=tuple(providers)),
        tiling=tiling,
        artifact_path=resolved_artifact,
    )


def default_model_root() -> Path:
    configured = os.environ.get("ARW_DENOISE_MODEL_DIR")
    if configured:
        return Path(configured).resolve()
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        return (Path(bundled_root) / "models").resolve()
    executable_models = Path(sys.executable).resolve().parent / "models"
    if getattr(sys, "frozen", False) or executable_models.is_dir():
        return executable_models.resolve()
    return (Path(__file__).resolve().parents[2] / "models").resolve()
