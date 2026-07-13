from __future__ import annotations

import numpy as np
import pytest

from arw_denoise.engines import CpuHaarEngine, DenoiseRequest


def test_cpu_engine_implements_common_result_contract() -> None:
    rng = np.random.default_rng(19)
    packed = rng.uniform(0.0, 1.0, (64, 80, 4)).astype(np.float32)
    original = packed.copy()

    result = CpuHaarEngine().run(DenoiseRequest(packed=packed, effective_iso=1600.0, strength=0.8))

    assert result.packed.shape == packed.shape
    assert result.packed.dtype == np.float32
    assert np.isfinite(result.packed).all()
    assert float(result.packed.min()) >= 0.0
    assert float(result.packed.max()) <= 1.0
    assert np.array_equal(packed, original)
    assert result.engine.engine_id == "cpu-haar"
    assert result.engine.provider == "CPU"
    assert not result.engine.is_gpu
    assert result.stats.inference_seconds >= 0.0


@pytest.mark.parametrize("strength", [-0.01, 2.01])
def test_common_request_rejects_invalid_strength(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        DenoiseRequest(
            packed=np.zeros((16, 16, 4), dtype=np.float32),
            effective_iso=100.0,
            strength=strength,
        ).validate()


def test_common_request_rejects_invalid_bayer_contract() -> None:
    with pytest.raises(ValueError, match="HxWx4"):
        DenoiseRequest(
            packed=np.zeros((16, 16, 3), dtype=np.float32),
            effective_iso=100.0,
        ).validate()

