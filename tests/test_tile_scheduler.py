from __future__ import annotations

import numpy as np
import pytest

from arw_denoise.engines import DenoiseRequest, DenoiseResult, EngineInfo, EngineRunStats
from arw_denoise.onnx_engine import GpuRuntimeError
from arw_denoise.task_control import CancellationToken, ProcessingCancelled
from arw_denoise.tile_scheduler import (
    AdaptiveTileRunner,
    GpuFallbackRequired,
    choose_tile_sizes,
    is_cuda_oom,
)


class RecordingEngine:
    def __init__(self, seen_shapes: list[tuple[int, int]], max_size: int | None = None, error: str | None = None):
        self.seen_shapes = seen_shapes
        self.max_size = max_size
        self.error = error

    def run(self, request: DenoiseRequest) -> DenoiseResult:
        height, width, _ = request.packed.shape
        self.seen_shapes.append((height, width))
        if self.error:
            raise GpuRuntimeError(self.error)
        if self.max_size is not None and max(height, width) > self.max_size:
            raise GpuRuntimeError("CUDA out of memory while allocating tensor")
        return DenoiseResult(
            packed=np.clip(request.packed * 0.9, 0.0, 1.0).astype(np.float32),
            engine=EngineInfo(
                engine_id="onnx-pmrid",
                display_name="PMRID",
                provider="CUDAExecutionProvider",
                is_gpu=True,
                model_id="pmrid-general-raw",
                model_version="1.0.0",
            ),
            stats=EngineRunStats(inference_seconds=0.01, tile_size=height),
        )


def test_choose_tile_sizes_uses_vram_and_finite_retry_sequence() -> None:
    assert choose_tile_sizes(8192, recommended_size=1024, minimum_size=256) == (1024, 768, 512, 384, 256)
    assert choose_tile_sizes(3072, recommended_size=1024, minimum_size=256)[0] == 512


def test_adaptive_runner_restarts_smaller_after_cuda_oom() -> None:
    seen_shapes: list[tuple[int, int]] = []
    factories = 0
    resets = 0

    def factory():
        nonlocal factories
        factories += 1
        return RecordingEngine(seen_shapes, max_size=512)

    def reset():
        nonlocal resets
        resets += 1

    runner = AdaptiveTileRunner(
        factory,
        memory_total_mb=8192,
        recommended_size=1024,
        minimum_size=256,
        overlap=64,
        reset_engine=reset,
    )
    packed = np.full((700, 700, 4), 0.5, dtype=np.float32)
    result = runner.run(DenoiseRequest(packed=packed, effective_iso=1600.0))

    assert result.packed.shape == packed.shape
    assert np.allclose(result.packed, 0.45, atol=1e-6)
    assert result.stats.tile_size == 512
    assert factories == 3
    assert resets == 2
    assert seen_shapes[:3] == [(704, 704), (704, 704), (512, 512)]


def test_adaptive_runner_pads_irregular_small_image_to_model_multiple() -> None:
    seen_shapes: list[tuple[int, int]] = []
    runner = AdaptiveTileRunner(
        lambda: RecordingEngine(seen_shapes),
        memory_total_mb=8192,
        recommended_size=1024,
        minimum_size=256,
        overlap=64,
    )
    packed = np.full((117, 131, 4), 0.5, dtype=np.float32)
    result = runner.run(DenoiseRequest(packed=packed, effective_iso=1600.0))
    assert result.packed.shape == packed.shape
    assert seen_shapes == [(128, 144)]


def test_adaptive_runner_does_not_retry_non_oom_errors() -> None:
    factories = 0

    def factory():
        nonlocal factories
        factories += 1
        return RecordingEngine([], error="invalid model output")

    runner = AdaptiveTileRunner(factory, memory_total_mb=8192, recommended_size=1024, minimum_size=256, overlap=64)
    with pytest.raises(GpuRuntimeError, match="invalid model"):
        runner.run(
            DenoiseRequest(
                packed=np.zeros((256, 256, 4), dtype=np.float32),
                effective_iso=1600.0,
            )
        )
    assert factories == 1


def test_adaptive_runner_requests_cpu_fallback_after_all_oom_sizes() -> None:
    runner = AdaptiveTileRunner(
        lambda: RecordingEngine([], max_size=1),
        memory_total_mb=3072,
        recommended_size=1024,
        minimum_size=256,
        overlap=64,
    )
    with pytest.raises(GpuFallbackRequired) as caught:
        runner.run(
            DenoiseRequest(
                packed=np.zeros((600, 600, 4), dtype=np.float32),
                effective_iso=1600.0,
            )
        )
    assert caught.value.attempted_sizes == (512, 384, 256)


def test_cuda_oom_detection_requires_cuda_context() -> None:
    assert is_cuda_oom(GpuRuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not is_cuda_oom(MemoryError("out of memory"))


def test_cancelled_oom_does_not_retry_smaller_tile_or_request_cpu_fallback() -> None:
    token = CancellationToken()
    factories = 0

    class CancellingOomEngine(RecordingEngine):
        def run(self, request):
            token.cancel()
            raise GpuRuntimeError("CUDA out of memory while allocating tensor")

    def factory():
        nonlocal factories
        factories += 1
        return CancellingOomEngine([])

    runner = AdaptiveTileRunner(
        factory,
        memory_total_mb=8192,
        recommended_size=1024,
        minimum_size=256,
        overlap=64,
    )
    with pytest.raises(ProcessingCancelled):
        runner.run(
            DenoiseRequest(np.zeros((600, 600, 4), np.float32), effective_iso=1600),
            cancellation=token,
        )
    assert factories == 1

