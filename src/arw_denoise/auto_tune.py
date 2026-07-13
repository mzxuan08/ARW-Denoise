from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .denoise import estimate_noise
from .domain import RawMetadata


AUTO_STRATEGY_VERSION = "pmrid-auto-v1"


@dataclass(frozen=True)
class AutoDenoiseConfig:
    strategy_version: str
    effective_iso: float
    strength: float
    chroma_noise: float
    detail_protection: float
    artifact_suppression: float
    noise_sigma: tuple[float, float, float, float]
    confidence: float


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _equivalent_noise_iso(sigma: float) -> float:
    # A normalized sigma near 0.01 maps to a conservative ISO 400 baseline.
    return _clamp(400.0 * (max(sigma, 1e-5) / 0.01) ** 2, 100.0, 25600.0)


def tune_automatic(packed: np.ndarray, metadata: RawMetadata | None = None) -> AutoDenoiseConfig:
    estimate = estimate_noise(packed)
    median_sigma = float(np.median(np.asarray(estimate.sigma, dtype=np.float32)))
    noise_iso = _equivalent_noise_iso(median_sigma)
    exif_iso = metadata.iso if metadata is not None else None
    if exif_iso is not None and exif_iso > 0:
        bounded_exif = _clamp(float(exif_iso), 100.0, 25600.0)
        # Blend in log space because ISO and shot noise are multiplicative scales.
        effective_iso = math.exp(0.65 * math.log(bounded_exif) + 0.35 * math.log(noise_iso))
    else:
        effective_iso = noise_iso
    effective_iso = _clamp(effective_iso, 100.0, 25600.0)

    stops = _clamp(math.log2(effective_iso / 100.0), 0.0, 8.0)
    confidence = _clamp(float(estimate.confidence), 0.0, 1.0)
    base_strength = 0.35 + 0.10 * stops
    strength = _clamp(base_strength * (0.70 + 0.30 * confidence), 0.20, 1.20)
    chroma_noise = _clamp(0.35 + 0.06 * stops, 0.35, 0.85)
    detail_protection = _clamp(0.82 - 0.025 * stops, 0.58, 0.82)
    shutter = metadata.shutter_seconds if metadata is not None else None
    long_exposure = 0.0
    if shutter is not None and shutter > 1.0:
        long_exposure = _clamp(math.log2(shutter + 1.0) / 8.0, 0.0, 0.25)
    artifact_suppression = _clamp(0.35 + 0.035 * stops + long_exposure, 0.35, 0.85)

    return AutoDenoiseConfig(
        strategy_version=AUTO_STRATEGY_VERSION,
        effective_iso=effective_iso,
        strength=strength,
        chroma_noise=chroma_noise,
        detail_protection=detail_protection,
        artifact_suppression=artifact_suppression,
        noise_sigma=estimate.sigma,
        confidence=confidence,
    )

