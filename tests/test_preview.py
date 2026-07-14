from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arw_denoise.preview import PreviewPair, render_preview_pair


class FakeRaw:
    camera_whitebalance = [2.0, 1.0, 1.5, 1.0]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_pair_uses_identical_white_balance_and_render_mode(tmp_path: Path) -> None:
    source = tmp_path / "a.ARW"
    output = tmp_path / "a.dng"
    source.write_bytes(b"raw")
    output.write_bytes(b"dng")
    module = SimpleNamespace(imread=lambda _path: FakeRaw())
    calls = []

    def renderer(path, **kwargs):
        calls.append((path, kwargs))
        return np.zeros((10, 12, 3), np.uint8)

    pair = render_preview_pair(source, output, rawpy_module=module, renderer=renderer)
    pair.validate()
    assert calls[0][1]["white_balance"] == calls[1][1]["white_balance"]
    assert calls[0][1]["half_size"] is True


def test_pair_rejects_mismatched_orientation_or_size() -> None:
    with pytest.raises(ValueError, match="方向和尺寸"):
        PreviewPair(
            np.zeros((10, 12, 3), np.uint8), np.zeros((12, 10, 3), np.uint8)
        ).validate()


def test_render_pair_center_crops_small_demosaic_border_difference(tmp_path: Path) -> None:
    source = tmp_path / "a.ARW"
    output = tmp_path / "a.dng"
    source.write_bytes(b"raw")
    output.write_bytes(b"dng")
    module = SimpleNamespace(imread=lambda _path: FakeRaw())

    def renderer(path, **_kwargs):
        height = 18 if path.suffix.lower() == ".arw" else 20
        return np.zeros((height, 12, 3), np.uint8)

    pair = render_preview_pair(source, output, rawpy_module=module, renderer=renderer)
    assert pair.source.shape == pair.denoised.shape == (18, 12, 3)
