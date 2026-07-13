from pathlib import Path

import numpy as np
import pytest

from arw_denoise.domain import RawMetadata, UnsupportedRawError
from arw_denoise.pipeline import pack_normalized_bayer, tiled_inference, unpack_normalized_bayer


def metadata() -> RawMetadata:
    return RawMetadata(
        path=Path("sample.ARW"), width=12, height=8, raw_width=12, raw_height=8,
        cfa_pattern=(0, 1, 1, 2), color_description="RGBG",
        black_levels=(512, 512, 512, 512), white_level=16383, bits_per_sample=14,
        make="Sony", model="ILCE-7CM2",
    )


def test_pack_unpack_is_lossless_for_unchanged_values():
    rng = np.random.default_rng(7)
    pixels = rng.integers(512, 16384, size=(8, 12), dtype=np.uint16)
    packed, context = pack_normalized_bayer(pixels, metadata())
    restored = unpack_normalized_bayer(packed, context)
    np.testing.assert_array_equal(restored, pixels)


def test_tiled_identity_has_no_seams():
    rng = np.random.default_rng(8)
    image = rng.random((73, 91, 4), dtype=np.float32)
    result = tiled_inference(image, lambda tile: tile, tile_size=32, overlap=8)
    np.testing.assert_allclose(result, image, atol=1e-6)


def test_tiled_inference_rejects_bad_shape():
    image = np.zeros((64, 64, 4), np.float32)
    with pytest.raises(ValueError):
        tiled_inference(image, lambda tile: tile[:, :-1], tile_size=32, overlap=8)


def test_tiled_inference_reports_progress():
    image = np.zeros((80, 80, 4), dtype=np.float32)
    progress = []
    tiled_inference(
        image,
        lambda tile: tile,
        tile_size=48,
        overlap=8,
        on_progress=lambda completed, total: progress.append((completed, total)),
    )
    assert progress[-1][0] == progress[-1][1]
    assert [item[0] for item in progress] == list(range(1, progress[-1][1] + 1))


def test_metadata_rejects_non_bayer_color_layout():
    bad = metadata()
    object.__setattr__(bad, "cfa_pattern", (0, 0, 1, 2))
    with pytest.raises(UnsupportedRawError, match="CFA"):
        bad.validate()
