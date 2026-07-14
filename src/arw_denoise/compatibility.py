from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .domain import ArwDenoiseError, RawMetadata
from .raw import RawDecoder, RawPyDecoder


@dataclass(frozen=True)
class CompatibilityResult:
    path: Path
    metadata: RawMetadata | None = None
    error: str | None = None

    @property
    def supported(self) -> bool:
        return self.metadata is not None and self.error is None

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "path": str(Path(self.path).resolve()),
            "supported": self.supported,
        }
        if self.metadata is not None:
            document["camera"] = source_snapshot(self.metadata)
        if self.error:
            document["error"] = self.error
        return document


def source_snapshot(metadata: RawMetadata) -> dict[str, object]:
    return {
        "make": metadata.make,
        "model": metadata.model,
        "iso": metadata.iso,
        "width": metadata.width,
        "height": metadata.height,
        "bits_per_sample": metadata.bits_per_sample,
        "cfa": "".join(metadata.resolved_cfa),
    }


def discover_arw_files(root: Path, *, recursive: bool = True) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"找不到扫描目录：{root}")
    candidates = root.rglob("*") if recursive else root.glob("*")
    files = [path.resolve() for path in candidates if path.is_file() and path.suffix.casefold() == ".arw"]
    return sorted(
        files,
        key=lambda path: (
            len(path.relative_to(root).parts),
            path.relative_to(root).as_posix().casefold(),
        ),
    )


def _scan_one(path: Path, decoder_factory: Callable[[], RawDecoder]) -> CompatibilityResult:
    resolved = Path(path).expanduser().resolve()
    try:
        metadata = decoder_factory().probe(resolved)
        metadata.validate()
        return CompatibilityResult(resolved, metadata=metadata)
    except (ArwDenoiseError, OSError, ValueError) as exc:
        return CompatibilityResult(resolved, error=str(exc))
    except Exception as exc:
        return CompatibilityResult(resolved, error=f"预检失败：{exc}")


def scan_arw_files(
    paths: Iterable[Path],
    *,
    decoder_factory: Callable[[], RawDecoder] = RawPyDecoder,
    max_workers: int = 2,
) -> list[CompatibilityResult]:
    if not 1 <= max_workers <= 8:
        raise ValueError("预检并发数必须在 1–8 之间")
    ordered = [Path(path) for path in paths]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="arw-preflight") as executor:
        return list(executor.map(lambda path: _scan_one(path, decoder_factory), ordered))


def write_compatibility_report(results: Iterable[CompatibilityResult], path: Path) -> Path:
    records = list(results)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "summary": {
            "total": len(records),
            "supported": sum(result.supported for result in records),
            "unsupported": sum(not result.supported for result in records),
        },
        "files": [result.to_document() for result in records],
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
