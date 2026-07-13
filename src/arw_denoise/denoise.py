from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseEstimate:
    sigma: tuple[float, float, float, float]
    confidence: float


def estimate_noise(packed: np.ndarray) -> NoiseEstimate:
    """Conservative blind estimate based on low-signal high-pass residuals."""
    if packed.ndim != 3 or packed.shape[-1] != 4:
        raise ValueError("packed Bayer 必须是 HxWx4")
    sigmas: list[float] = []
    confidences: list[float] = []
    for channel in range(4):
        plane = packed[:, :, channel].astype(np.float32)
        threshold = float(np.quantile(plane, 0.35))
        core = plane[1:-1, 1:-1]
        residual = core - 0.25 * (
            plane[:-2, 1:-1] + plane[2:, 1:-1] + plane[1:-1, :-2] + plane[1:-1, 2:]
        )
        samples = residual[core <= threshold]
        if samples.size < 128:
            samples = residual.reshape(-1)
        median = float(np.median(samples))
        sigma = 1.4826 * float(np.median(np.abs(samples - median)))
        sigmas.append(max(sigma, 1e-5))
        confidences.append(min(1.0, samples.size / 4096.0))
    return NoiseEstimate(tuple(sigmas), float(min(confidences)))  # type: ignore[arg-type]


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _haar_shrink(plane: np.ndarray, threshold: float) -> np.ndarray:
    original_h, original_w = plane.shape
    h = original_h - original_h % 2
    w = original_w - original_w % 2
    core = plane[:h, :w]
    a = core[0::2, 0::2]
    b = core[0::2, 1::2]
    c = core[1::2, 0::2]
    d = core[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = _soft_threshold((a - b + c - d) * 0.5, threshold)
    hl = _soft_threshold((a + b - c - d) * 0.5, threshold)
    hh = _soft_threshold((a - b - c + d) * 0.5, threshold * 1.15)
    out = plane.copy()
    out[0:h:2, 0:w:2] = (ll + lh + hl + hh) * 0.5
    out[0:h:2, 1:w:2] = (ll - lh + hl - hh) * 0.5
    out[1:h:2, 0:w:2] = (ll + lh - hl - hh) * 0.5
    out[1:h:2, 1:w:2] = (ll - lh - hl + hh) * 0.5
    return out


class HaarWaveletDenoiser:
    """Small dependency-free CPU baseline; intentionally conservative."""

    def denoise(self, packed: np.ndarray, strength: float = 1.0) -> np.ndarray:
        if not 0.0 <= strength <= 2.0:
            raise ValueError("strength 必须在 0–2")
        estimate = estimate_noise(packed)
        result = packed.astype(np.float32).copy()
        confidence_scale = 0.65 + 0.35 * estimate.confidence
        for channel, sigma in enumerate(estimate.sigma):
            threshold = sigma * (0.8 + 0.9 * strength) * confidence_scale
            result[:, :, channel] = _haar_shrink(result[:, :, channel], threshold)
        return np.clip(result, 0.0, 1.0)

