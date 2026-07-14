from __future__ import annotations

from pathlib import Path

import pytest

from arw_denoise.cache_storage import (
    cache_usage,
    clear_managed_cache,
    ensure_managed_cache,
)


def test_managed_cache_can_be_cleared_without_touching_siblings(tmp_path: Path) -> None:
    cache = tmp_path / "selected-parent" / "ArwDenoiseCache"
    sibling = tmp_path / "selected-parent" / "keep.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"keep")
    ensure_managed_cache(cache)
    (cache / "preview.npy").write_bytes(b"1234")
    nested = cache / "nested"
    nested.mkdir()
    (nested / "temporary.bin").write_bytes(b"567")

    assert cache_usage(cache) == (2, 7)
    removed_files, removed_bytes = clear_managed_cache(cache)

    assert (removed_files, removed_bytes) == (2, 7)
    assert cache_usage(cache) == (0, 0)
    assert sibling.read_bytes() == b"keep"


def test_clear_refuses_a_directory_not_created_by_the_application(tmp_path: Path) -> None:
    unmanaged = tmp_path / "important"
    unmanaged.mkdir()
    (unmanaged / "photo.ARW").write_bytes(b"raw")

    with pytest.raises(ValueError, match="不是 ARW Denoise 管理的缓存目录"):
        clear_managed_cache(unmanaged)

    assert (unmanaged / "photo.ARW").read_bytes() == b"raw"


def test_manager_refuses_to_adopt_a_nonempty_custom_directory(tmp_path: Path) -> None:
    custom_cache = tmp_path / "ArwDenoiseCache"
    custom_cache.mkdir()
    (custom_cache / "important.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="包含其他文件"):
        ensure_managed_cache(custom_cache)

    assert (custom_cache / "important.txt").read_text(encoding="utf-8") == "keep"
