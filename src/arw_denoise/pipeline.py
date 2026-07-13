from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .domain import RawMetadata, UnsupportedRawError


@dataclass(frozen=True)
class NormalizationContext:
    black_by_position: np.ndarray
    range_by_position: np.ndarray
    dtype: np.dtype


def _position_values(values_by_color: tuple[float, float, float, float], pattern: tuple[int, int, int, int]) -> np.ndarray:
    values = np.asarray(values_by_color, dtype=np.float32)
    return values[np.asarray(pattern, dtype=np.intp)]


def pack_normalized_bayer(pixels: np.ndarray, metadata: RawMetadata) -> tuple[np.ndarray, NormalizationContext]:
    metadata.validate()
    if pixels.shape != (metadata.height, metadata.width):
        raise UnsupportedRawError("Bayer 数组尺寸与元数据不一致")
    if not np.issubdtype(pixels.dtype, np.integer):
        raise UnsupportedRawError("输入 Bayer 数据必须是整数采样")

    black = _position_values(metadata.black_levels, metadata.cfa_pattern)
    white = np.full(4, metadata.white_level, dtype=np.float32)
    ranges = white - black
    if np.any(ranges <= 0):
        raise UnsupportedRawError("存在无效的 Bayer 黑白电平范围")

    packed = np.stack(
        (pixels[0::2, 0::2], pixels[0::2, 1::2], pixels[1::2, 0::2], pixels[1::2, 1::2]),
        axis=-1,
    ).astype(np.float32)
    packed = np.clip((packed - black.reshape(1, 1, 4)) / ranges.reshape(1, 1, 4), 0.0, 1.0)
    context = NormalizationContext(black, ranges, pixels.dtype)
    return packed, context


def unpack_normalized_bayer(packed: np.ndarray, context: NormalizationContext) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[-1] != 4:
        raise ValueError("packed Bayer 必须是 HxWx4")
    if not np.all(np.isfinite(packed)):
        raise ValueError("降噪结果包含 NaN 或无穷值")
    packed = np.clip(packed.astype(np.float32), 0.0, 1.0)
    restored = packed * context.range_by_position.reshape(1, 1, 4) + context.black_by_position.reshape(1, 1, 4)
    height, width, _ = restored.shape
    output = np.empty((height * 2, width * 2), dtype=context.dtype)
    limits = np.iinfo(context.dtype)
    restored = np.rint(np.clip(restored, limits.min, limits.max)).astype(context.dtype)
    output[0::2, 0::2] = restored[:, :, 0]
    output[0::2, 1::2] = restored[:, :, 1]
    output[1::2, 0::2] = restored[:, :, 2]
    output[1::2, 1::2] = restored[:, :, 3]
    return output


def tiled_inference(
    image: np.ndarray,
    infer: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 1024,
    overlap: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Run HWC inference with overlap-add blending and no hard seams."""
    if image.ndim != 3:
        raise ValueError("image 必须是 HxWxC")
    if tile_size <= overlap * 2:
        raise ValueError("tile_size 必须大于两倍 overlap")
    height, width, channels = image.shape
    if height <= tile_size and width <= tile_size:
        result = infer(image)
        if result.shape != image.shape:
            raise ValueError("模型输出尺寸与输入不一致")
        if on_progress:
            on_progress(1, 1)
        return result

    step = tile_size - overlap
    output = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    ramp = np.linspace(0.05, 1.0, overlap, dtype=np.float32) if overlap else np.ones(1, np.float32)

    def starts(length: int) -> list[int]:
        values: list[int] = []
        for position in range(0, length, step):
            adjusted = min(position, max(0, length - tile_size))
            if not values or adjusted != values[-1]:
                values.append(adjusted)
            if adjusted + tile_size >= length:
                break
        return values

    top_positions = starts(height)
    left_positions = starts(width)
    total = len(top_positions) * len(left_positions)
    completed = 0
    for top in top_positions:
        bottom = min(height, top + tile_size)
        for left in left_positions:
            right = min(width, left + tile_size)
            tile = image[top:bottom, left:right]
            prediction = infer(tile)
            if prediction.shape != tile.shape:
                raise ValueError("模型输出尺寸与输入分块不一致")
            window = np.ones((bottom - top, right - left), dtype=np.float32)
            if overlap:
                if top > 0:
                    window[:overlap, :] *= ramp[:, None]
                if bottom < height:
                    window[-overlap:, :] *= ramp[::-1, None]
                if left > 0:
                    window[:, :overlap] *= ramp[None, :]
                if right < width:
                    window[:, -overlap:] *= ramp[None, ::-1]
            output[top:bottom, left:right] += prediction.astype(np.float32) * window[:, :, None]
            weights[top:bottom, left:right] += window[:, :, None]
            completed += 1
            if on_progress:
                on_progress(completed, total)
    return output / np.maximum(weights, 1e-8)
