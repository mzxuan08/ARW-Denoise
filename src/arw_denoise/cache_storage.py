from __future__ import annotations

import shutil
from pathlib import Path


CACHE_MARKER = ".arw-denoise-cache"


def _resolved_cache_root(root: Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("不能把磁盘根目录直接作为缓存目录")
    return resolved


def ensure_managed_cache(root: Path, *, adopt_existing: bool = False) -> Path:
    resolved = _resolved_cache_root(root)
    resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / CACHE_MARKER
    if marker.exists() and not marker.is_file():
        raise ValueError(f"缓存标记无效：{marker}")
    if not marker.exists() and any(resolved.iterdir()) and not adopt_existing:
        raise ValueError(f"缓存目录已存在且包含其他文件：{resolved}")
    marker.write_text("ARW Denoise managed cache\n", encoding="utf-8")
    return resolved


def _require_managed_cache(root: Path) -> Path:
    resolved = _resolved_cache_root(root)
    if not resolved.is_dir() or not (resolved / CACHE_MARKER).is_file():
        raise ValueError(f"不是 ARW Denoise 管理的缓存目录：{resolved}")
    return resolved


def cache_usage(root: Path) -> tuple[int, int]:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        return 0, 0
    files = [
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.name != CACHE_MARKER
    ]
    return len(files), sum(path.stat().st_size for path in files)


def clear_managed_cache(root: Path) -> tuple[int, int]:
    resolved = _require_managed_cache(root)
    file_count, byte_count = cache_usage(resolved)
    for child in resolved.iterdir():
        if child.name == CACHE_MARKER:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    return file_count, byte_count
