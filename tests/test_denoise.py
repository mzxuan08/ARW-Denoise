import numpy as np

from arw_denoise.denoise import HaarWaveletDenoiser, estimate_noise


def test_noise_estimator_and_wavelet_reduce_flat_field_variance():
    rng = np.random.default_rng(42)
    clean = np.full((128, 128, 4), 0.2, dtype=np.float32)
    noisy = np.clip(clean + rng.normal(0, 0.025, clean.shape), 0, 1).astype(np.float32)
    estimate = estimate_noise(noisy)
    assert all(value > 0 for value in estimate.sigma)
    result = HaarWaveletDenoiser().denoise(noisy, strength=1.0)
    assert float(np.var(result - clean)) < float(np.var(noisy - clean))
    assert abs(float(np.mean(result)) - 0.2) < 0.01

