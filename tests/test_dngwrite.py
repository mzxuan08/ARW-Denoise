from pathlib import Path

import numpy as np
import tifffile
import pytest

from arw_denoise.dngwrite import replace_cfa_pixels_in_place
from arw_denoise.domain import RawMetadata
from arw_denoise.domain import ExternalToolError
from arw_denoise.dngwrite import validate_processed_dng


def test_replace_cfa_pixels_preserves_masked_border(tmp_path: Path):
    path = tmp_path / "base.dng"
    full = np.full((10, 14), 99, dtype=np.uint16)
    tifffile.imwrite(
        path,
        full,
        photometric=32803,
        compression=None,
        metadata=None,
        extratags=[
            (33421, "H", 2, (2, 2), False),
            (33422, "B", 4, (0, 1, 1, 2), False),
            (50714, "H", 4, (512, 512, 512, 512), False),
            (50717, "H", 1, 16383, False),
            (50829, "I", 4, (2, 2, 8, 12), False),
        ],
    )
    metadata = RawMetadata(
        path=Path("sample.ARW"), width=10, height=6, raw_width=14, raw_height=10,
        cfa_pattern=(0, 1, 1, 2), color_description="RGBG",
        black_levels=(512, 512, 512, 512), white_level=16383, bits_per_sample=14,
        top_margin=2, left_margin=2,
    )
    visible = np.arange(60, dtype=np.uint16).reshape(6, 10)
    replace_cfa_pixels_in_place(path, visible, metadata)
    result = tifffile.imread(path)
    np.testing.assert_array_equal(result[2:8, 2:12], visible)
    assert np.all(result[:2] == 99)
    assert np.all(result[:, :2] == 99)


def test_validation_rejects_wrong_pixels(tmp_path: Path):
    path = tmp_path / "base.dng"
    pixels = np.full((8, 12), 512, np.uint16)
    tifffile.imwrite(
        path, pixels, photometric=32803, compression=None, metadata=None,
        extratags=[
            (33421, "H", 2, (2, 2), False), (33422, "B", 4, (0, 1, 1, 2), False),
            (50714, "H", 4, (512, 512, 512, 512), False), (50717, "H", 1, 16383, False),
        ],
    )
    metadata = RawMetadata(
        path=Path("sample.ARW"), width=12, height=8, raw_width=12, raw_height=8,
        cfa_pattern=(0, 1, 1, 2), color_description="RGBG",
        black_levels=(512, 512, 512, 512), white_level=16383, bits_per_sample=14,
    )
    with pytest.raises(ExternalToolError, match="像素回读"):
        validate_processed_dng(path, pixels + 1, metadata)
