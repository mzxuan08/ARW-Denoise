from __future__ import annotations

import pytest

from arw_denoise.eta import DurationSample, EtaEstimator


def test_eta_requires_two_matching_engine_samples() -> None:
    estimator = EtaEstimator()
    estimator.add(DurationSample("gpu", 24_000_000, 12.0))
    assert estimator.estimate("gpu", [24_000_000]) is None
    assert estimator.estimate("cpu", [24_000_000]) is None


def test_eta_uses_median_seconds_per_pixel_and_pending_sizes() -> None:
    estimator = EtaEstimator()
    estimator.add(DurationSample("gpu", 20_000_000, 10.0))
    estimator.add(DurationSample("gpu", 40_000_000, 24.0))
    estimator.add(DurationSample("gpu", 20_000_000, 200.0))  # rejected outlier
    result = estimator.estimate("gpu", [10_000_000, 30_000_000])
    assert result == pytest.approx(22.0)


def test_eta_rejects_invalid_samples_and_caps_history() -> None:
    estimator = EtaEstimator(max_samples=3)
    with pytest.raises(ValueError):
        estimator.add(DurationSample("gpu", 0, 1.0))
    with pytest.raises(ValueError):
        estimator.add(DurationSample("gpu", 1, -1.0))
    for seconds in (1.0, 2.0, 3.0, 4.0):
        estimator.add(DurationSample("gpu", 1_000_000, seconds))
    assert estimator.sample_count("gpu") == 3
