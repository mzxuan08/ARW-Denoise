from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .engines import DenoiseRequest, DenoiseResult, EngineInfo, EngineRunStats, RawDenoiseEngine
from .onnx_engine import GpuRuntimeError
from .pipeline import tiled_inference


class GpuFallbackRequired(GpuRuntimeError):
    def __init__(self, attempted_sizes: tuple[int, ...], last_error: BaseException):
        self.attempted_sizes = attempted_sizes
        self.last_error = last_error
        sizes = "、".join(str(value) for value in attempted_sizes)
        super().__init__(f"CUDA 在分块 {sizes} 均显存不足，需要回退 CPU：{last_error}")


def _exception_messages(error: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(messages).lower()


def is_cuda_oom(error: BaseException) -> bool:
    message = _exception_messages(error)
    cuda_context = any(token in message for token in ("cuda", "cudnn", "cublas", "gpu"))
    oom = any(
        token in message
        for token in (
            "out of memory",
            "error_out_of_memory",
            "failed to allocate memory",
            "failed to allocate buffer",
        )
    )
    return cuda_context and oom


def choose_tile_sizes(
    memory_total_mb: int | None,
    *,
    recommended_size: int,
    minimum_size: int,
) -> tuple[int, ...]:
    if recommended_size < minimum_size or minimum_size < 64:
        raise ValueError("模型分块范围无效")
    if memory_total_mb is None or memory_total_mb >= 6144:
        initial = recommended_size
    elif memory_total_mb >= 4096:
        initial = min(recommended_size, 768)
    else:
        initial = min(recommended_size, 512)
    candidates = (initial, 768, 512, 384, minimum_size)
    sizes: list[int] = []
    for value in candidates:
        aligned = value - value % 16
        if minimum_size <= aligned <= initial and aligned not in sizes:
            sizes.append(aligned)
    if minimum_size not in sizes:
        sizes.append(minimum_size)
    return tuple(sizes)


def _pad_model_tile(tile: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int, int]]:
    height, width, _ = tile.shape
    padded_height = ((height + multiple - 1) // multiple) * multiple
    padded_width = ((width + multiple - 1) // multiple) * multiple
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if not pad_bottom and not pad_right:
        return tile, (height, width)
    mode = "reflect" if height > 1 and width > 1 else "edge"
    padded = np.pad(tile, ((0, pad_bottom), (0, pad_right), (0, 0)), mode=mode)
    return np.ascontiguousarray(padded, dtype=np.float32), (height, width)


class AdaptiveTileRunner:
    def __init__(
        self,
        engine_factory: Callable[[], RawDenoiseEngine],
        *,
        memory_total_mb: int | None,
        recommended_size: int,
        minimum_size: int,
        overlap: int,
    ):
        if overlap < 0 or minimum_size <= 2 * overlap:
            raise ValueError("overlap 对最小分块过大")
        self.engine_factory = engine_factory
        self.memory_total_mb = memory_total_mb
        self.recommended_size = recommended_size
        self.minimum_size = minimum_size
        self.overlap = overlap

    def run(
        self,
        request: DenoiseRequest,
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> DenoiseResult:
        request.validate()
        sizes = choose_tile_sizes(
            self.memory_total_mb,
            recommended_size=self.recommended_size,
            minimum_size=self.minimum_size,
        )
        attempted: list[int] = []
        last_oom: BaseException | None = None
        for tile_size in sizes:
            attempted.append(tile_size)
            engine = self.engine_factory()
            inference_seconds = 0.0
            engine_info: EngineInfo | None = None

            def infer(tile: np.ndarray) -> np.ndarray:
                nonlocal inference_seconds, engine_info
                padded, original_shape = _pad_model_tile(tile)
                tile_result = engine.run(
                    DenoiseRequest(
                        packed=padded,
                        effective_iso=request.effective_iso,
                        strength=request.strength,
                    )
                )
                inference_seconds += tile_result.stats.inference_seconds
                engine_info = tile_result.engine
                height, width = original_shape
                return tile_result.packed[:height, :width]

            try:
                packed = tiled_inference(
                    request.packed,
                    infer,
                    tile_size=tile_size,
                    overlap=self.overlap,
                    on_progress=on_progress,
                )
            except Exception as exc:
                if not is_cuda_oom(exc):
                    raise
                last_oom = exc
                del engine
                gc.collect()
                continue
            if engine_info is None:
                raise GpuRuntimeError("GPU 分块调度未产生结果")
            return DenoiseResult(
                packed=np.clip(packed, 0.0, 1.0).astype(np.float32, copy=False),
                engine=engine_info,
                stats=EngineRunStats(
                    inference_seconds=inference_seconds,
                    tile_size=tile_size,
                    peak_vram_mb=None,
                ),
            )
        assert last_oom is not None
        raise GpuFallbackRequired(tuple(attempted), last_oom)

