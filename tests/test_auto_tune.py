from __future__ import annotations

from pathlib import Path

import numpy as np

from arw_denoise.auto_tune import AUTO_STRATEGY_VERSION, tune_automatic
from arw_denoise.domain import RawMetadata


def _metadata(iso: int | None, shutter: float | None = 1 / 125) -> RawMetadata:
    return RawMetadata(
        path=Path("sample.ARW"),
        width=256,
        height=256,
        raw_width=256,
        raw_height=256,
        cfa_pattern=(0, 1, 3, 2),
        color_description="RGBG",
        black_levels=(512, 512, 512, 512),
        white_level=15360,
        bits_per_sample=14,
        make="Sony",
        model="ILCE-7CM2",
        iso=iso,
        shutter_seconds=shutter,
    )


def _noisy(sigma: float, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(123)
    clean = np.full((size, size, 4), 0.12, dtype=np.float32)
    return np.clip(clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0).astype(np.float32)


def test_auto_tune_increases_equivalent_iso_and_strength_with_noise() -> None:
    low = tune_automatic(_noisy(0.005), _metadata(None))
    high = tune_automatic(_noisy(0.04), _metadata(None))
    assert high.effective_iso > low.effective_iso
    assert high.strength > low.strength
    assert high.chroma_noise >= low.chroma_noise


def test_auto_tune_combines_exif_iso_without_exceeding_model_bounds() -> None:
    low = tune_automatic(_noisy(0.015), _metadata(100))
    high = tune_automatic(_noisy(0.015), _metadata(102400))
    assert 100.0 <= low.effective_iso <= 25600.0
    assert 100.0 <= high.effective_iso <= 25600.0
    assert high.effective_iso > low.effective_iso


def test_auto_tune_is_conservative_when_estimate_confidence_is_low() -> None:
    small = tune_automatic(_noisy(0.02, size=12), _metadata(1600))
    large = tune_automatic(_noisy(0.02, size=128), _metadata(1600))
    assert small.confidence < large.confidence
    assert small.strength < large.strength


def test_long_exposure_increases_artifact_suppression_not_exposure() -> None:
    normal = tune_automatic(_noisy(0.02), _metadata(1600, 1 / 125))
    long = tune_automatic(_noisy(0.02), _metadata(1600, 10.0))
    assert long.artifact_suppression > normal.artifact_suppression
    assert long.effective_iso == normal.effective_iso


def test_auto_tune_records_version_and_safe_ranges() -> None:
    config = tune_automatic(_noisy(0.03), _metadata(6400))
    assert config.strategy_version == AUTO_STRATEGY_VERSION
    assert 0.0 <= config.strength <= 2.0
    assert 0.0 <= config.chroma_noise <= 1.0
    assert 0.0 <= config.detail_protection <= 1.0
    assert 0.0 <= config.artifact_suppression <= 1.0

