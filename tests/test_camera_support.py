from __future__ import annotations

from pathlib import Path

import pytest

from arw_denoise.domain import RawMetadata, UnsupportedRawError
from arw_denoise.raw import validate_camera_support


def metadata(*, make: str | None = "SONY", model: str | None = "ILCE-7CM2") -> RawMetadata:
    return RawMetadata(
        path=Path("sample.ARW"),
        width=12,
        height=8,
        raw_width=12,
        raw_height=8,
        cfa_pattern=(0, 1, 1, 2),
        color_description="RGBG",
        black_levels=(512, 512, 512, 512),
        white_level=16383,
        bits_per_sample=14,
        make=make,
        model=model,
    )


@pytest.mark.parametrize(
    "model",
    [
        "ILCE-1",
        "ILCE-9M3",
        "ILCE-7RM5",
        "ILCE-7CM2",
        "ILCE-6700",
        "ILCE-6000",
        "NEX-7",
        "ZV-E1",
        "ILME-FX3",
        "DSC-RX100M7",
        "SLT-A99V",
        "DSLR-A900",
    ],
)
def test_all_sony_camera_families_are_accepted_by_capability(model: str) -> None:
    info = metadata(model=model)
    validate_camera_support(info)


@pytest.mark.parametrize("make", ["Sony", "SONY", "Sony Corporation"])
def test_sony_make_variants_are_accepted(make: str) -> None:
    validate_camera_support(metadata(make=make, model="UNKNOWN-FUTURE-MODEL"))


@pytest.mark.parametrize("make", [None, "", "Canon", "NIKON CORPORATION", "NotSony"])
def test_non_sony_or_missing_make_is_rejected_by_default(make: str | None) -> None:
    with pytest.raises(UnsupportedRawError, match="Sony ARW"):
        validate_camera_support(metadata(make=make))


def test_experimental_mode_can_still_bypass_brand_gate() -> None:
    validate_camera_support(metadata(make="Other", model="Prototype"), allow_experimental=True)


def test_camera_gate_still_rejects_unsupported_cfa_capability() -> None:
    info = metadata(model="FUTURE-SONY")
    object.__setattr__(info, "cfa_pattern", (0, 0, 1, 2))
    with pytest.raises(UnsupportedRawError, match="CFA"):
        validate_camera_support(info)
