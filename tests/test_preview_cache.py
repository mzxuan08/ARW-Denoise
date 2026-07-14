from __future__ import annotations

from pathlib import Path

import numpy as np

from arw_denoise.preview import PreviewPair
from arw_denoise.preview_cache import PreviewCache


def _files(tmp_path: Path) -> tuple[Path, Path]:
    source, output = tmp_path / "a.ARW", tmp_path / "a.dng"
    source.write_bytes(b"raw")
    output.write_bytes(b"dng")
    return source, output


def test_cache_hits_and_invalidates_when_input_changes(tmp_path: Path) -> None:
    source, output = _files(tmp_path)
    calls = 0

    def render(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        value = np.full((8, 10, 3), calls, np.uint8)
        return PreviewPair(value, value.copy())

    cache = PreviewCache(tmp_path / "cache")
    assert cache.get_or_render(source, output, renderer=render).source[0, 0, 0] == 1
    assert cache.get_or_render(source, output, renderer=render).source[0, 0, 0] == 1
    output.write_bytes(b"changed dng")
    assert cache.get_or_render(source, output, renderer=render).source[0, 0, 0] == 2


def test_corrupt_cache_is_rebuilt_and_lru_is_bounded(tmp_path: Path) -> None:
    source, output = _files(tmp_path)
    calls = 0

    def render(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        value = np.full((20, 20, 3), calls, np.uint8)
        return PreviewPair(value, value.copy())

    cache = PreviewCache(tmp_path / "cache", max_bytes=10000)
    cache.get_or_render(source, output, renderer=render)
    cached = next((tmp_path / "cache").glob("preview-*.npy"))
    cached.write_bytes(b"broken")
    cache.get_or_render(source, output, renderer=render)
    assert calls == 2
    cache.max_bytes = 1500
    assert cache.prune() == 2
    assert not list((tmp_path / "cache").glob("preview-*.npy"))
