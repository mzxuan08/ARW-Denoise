from __future__ import annotations

import numpy as np
import pytest

from arw_denoise.postprocess import PostprocessSettings, postprocess_raw
from arw_denoise.task_control import CancellationToken, ProcessingCancelled


def test_identity_model_output_remains_exactly_unchanged() -> None:
    rng = np.random.default_rng(5)
    original = rng.uniform(0.05, 0.9, (64, 64, 4)).astype(np.float32)
    result = postprocess_raw(original, original.copy(), PostprocessSettings())
    np.testing.assert_array_equal(result, original)


def test_strength_zero_returns_original_and_higher_strength_reduces_noise() -> None:
    rng = np.random.default_rng(6)
    clean = np.full((64, 64, 4), 0.3, dtype=np.float32)
    noisy = np.clip(clean + rng.normal(0.0, 0.03, clean.shape), 0.0, 1.0).astype(np.float32)
    off = postprocess_raw(noisy, clean, PostprocessSettings(strength=0.0))
    normal = postprocess_raw(noisy, clean, PostprocessSettings(strength=1.0))
    np.testing.assert_array_equal(off, noisy)
    assert float(np.mean((normal - clean) ** 2)) < float(np.mean((noisy - clean) ** 2))


def test_detail_protection_reduces_model_blur_at_strong_edge() -> None:
    original = np.zeros((64, 64, 4), dtype=np.float32)
    original[:, 32:, :] = 0.8
    blurred = original.copy()
    blurred[:, 30:34, :] = np.linspace(0.2, 0.6, 4, dtype=np.float32)[None, :, None]
    unprotected = postprocess_raw(
        original,
        blurred,
        PostprocessSettings(detail_protection=0.0, artifact_suppression=0.0),
    )
    protected = postprocess_raw(
        original,
        blurred,
        PostprocessSettings(detail_protection=1.0, artifact_suppression=0.0),
    )
    edge = np.s_[:, 30:34, :]
    assert float(np.mean(np.abs(protected[edge] - original[edge]))) < float(
        np.mean(np.abs(unprotected[edge] - original[edge]))
    )


def test_chroma_control_reduces_red_blue_residual_disagreement() -> None:
    original = np.full((48, 48, 4), 0.3, dtype=np.float32)
    model = original.copy()
    checker = (np.indices((48, 48)).sum(axis=0) % 2).astype(np.float32) * 0.08 - 0.04
    model[:, :, 0] += checker
    model[:, :, 3] -= checker
    off = postprocess_raw(original, model, PostprocessSettings(chroma_noise=0.0, artifact_suppression=0.0))
    on = postprocess_raw(original, model, PostprocessSettings(chroma_noise=1.0, artifact_suppression=0.0))
    assert float(np.var(on[:, :, 0] - on[:, :, 3])) < float(np.var(off[:, :, 0] - off[:, :, 3]))


def test_artifact_suppression_limits_isolated_model_delta() -> None:
    original = np.full((64, 64, 4), 0.25, dtype=np.float32)
    model = original.copy()
    model[32, 32, :] = 0.9
    off = postprocess_raw(original, model, PostprocessSettings(artifact_suppression=0.0))
    on = postprocess_raw(original, model, PostprocessSettings(artifact_suppression=1.0))
    assert float(np.max(np.abs(on[32, 32] - original[32, 32]))) < float(
        np.max(np.abs(off[32, 32] - original[32, 32]))
    )


def test_postprocess_preserves_channel_means_and_safe_range() -> None:
    rng = np.random.default_rng(7)
    original = rng.uniform(0.1, 0.7, (96, 96, 4)).astype(np.float32)
    model = np.clip(original * 0.9 + 0.03, 0.0, 1.0)
    result = postprocess_raw(original, model, PostprocessSettings(strength=1.2))
    np.testing.assert_allclose(result.mean(axis=(0, 1)), original.mean(axis=(0, 1)), atol=2e-5)
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_postprocess_can_reuse_model_output_without_touching_original() -> None:
    rng = np.random.default_rng(21)
    original = rng.uniform(0.1, 0.8, (40, 48, 4)).astype(np.float32)
    model = np.clip(original * 0.92 + 0.02, 0.0, 1.0).astype(np.float32)
    expected = postprocess_raw(original, model, PostprocessSettings())
    original_before = original.copy()
    result = postprocess_raw(original, model, PostprocessSettings(), out=model)
    assert result is model
    np.testing.assert_allclose(result, expected, atol=2e-7)
    np.testing.assert_array_equal(original, original_before)


@pytest.mark.parametrize(
    "settings",
    [
        PostprocessSettings(strength=-0.1),
        PostprocessSettings(chroma_noise=1.1),
        PostprocessSettings(detail_protection=-0.1),
        PostprocessSettings(artifact_suppression=1.1),
    ],
)
def test_postprocess_rejects_invalid_settings(settings: PostprocessSettings) -> None:
    with pytest.raises(ValueError):
        postprocess_raw(np.zeros((8, 8, 4), np.float32), np.zeros((8, 8, 4), np.float32), settings)


def test_postprocess_checks_cancellation_between_stages() -> None:
    token = CancellationToken()
    checks = 0
    original_check = token.check

    def check() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            token.cancel()
        original_check()

    token.check = check  # type: ignore[method-assign]
    original = np.full((64, 64, 4), 0.3, np.float32)
    prediction = np.full((64, 64, 4), 0.28, np.float32)
    with pytest.raises(ProcessingCancelled):
        postprocess_raw(original, prediction, PostprocessSettings(), cancellation=token)
    assert checks == 3

