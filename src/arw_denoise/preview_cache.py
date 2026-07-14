from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from .preview import PREVIEW_RENDER_VERSION, PreviewPair, render_preview_pair


def preview_identity(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class PreviewCache:
    def __init__(self, root: Path, *, max_bytes: int = 2 * 1024**3) -> None:
        if max_bytes <= 0:
            raise ValueError("预览缓存上限必须为正数")
        self.root = Path(root)
        self.max_bytes = max_bytes

    def _key(self, source: Path, denoised: Path, half_size: bool) -> str:
        payload = {
            "source": preview_identity(source),
            "denoised": preview_identity(denoised),
            "half_size": half_size,
            "renderer": PREVIEW_RENDER_VERSION,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"preview-{key}-source.npy", self.root / f"preview-{key}-denoised.npy"

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("预览缓存格式无效")
        return np.ascontiguousarray(value)

    @staticmethod
    def _save_atomic(path: Path, value: np.ndarray) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def get_or_render(
        self,
        source: Path,
        denoised: Path,
        *,
        half_size: bool = True,
        renderer: Callable[..., PreviewPair] = render_preview_pair,
    ) -> PreviewPair:
        self.root.mkdir(parents=True, exist_ok=True)
        key = self._key(source, denoised, half_size)
        source_path, denoised_path = self._paths(key)
        if source_path.is_file() and denoised_path.is_file():
            try:
                pair = PreviewPair(self._load(source_path), self._load(denoised_path))
                pair.validate()
                now = time.time()
                os.utime(source_path, (now, now))
                os.utime(denoised_path, (now, now))
                return pair
            except (OSError, ValueError):
                source_path.unlink(missing_ok=True)
                denoised_path.unlink(missing_ok=True)
        pair = renderer(Path(source), Path(denoised), half_size=half_size)
        pair.validate()
        self._save_atomic(source_path, pair.source)
        self._save_atomic(denoised_path, pair.denoised)
        self.prune()
        return pair

    def invalidate(self, source: Path, denoised: Path, *, half_size: bool) -> None:
        key = self._key(source, denoised, half_size)
        for path in self._paths(key):
            path.unlink(missing_ok=True)

    def prune(self) -> int:
        groups: dict[str, list[Path]] = {}
        for path in self.root.glob("preview-*.npy"):
            key = path.name.removeprefix("preview-").split("-", 1)[0]
            groups.setdefault(key, []).append(path)
        ordered = sorted(
            groups.values(),
            key=lambda paths: max(path.stat().st_mtime for path in paths),
            reverse=True,
        )
        total = 0
        removed = 0
        for paths in ordered:
            size = sum(path.stat().st_size for path in paths)
            if len(paths) == 2 and total + size <= self.max_bytes:
                total += size
            else:
                for path in paths:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed
