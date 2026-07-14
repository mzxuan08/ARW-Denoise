from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .domain import RawMetadata, UnsupportedRawError
from .task_control import CancellationToken


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

    packed = np.empty((metadata.height // 2, metadata.width // 2, 4), dtype=np.float32)
    for position, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        packed[:, :, position] = pixels[row::2, column::2]
        np.subtract(packed[:, :, position], black[position], out=packed[:, :, position])
        np.divide(packed[:, :, position], ranges[position], out=packed[:, :, position])
    np.clip(packed, 0.0, 1.0, out=packed)
    context = NormalizationContext(black, ranges, pixels.dtype)
    return packed, context


def unpack_normalized_bayer(
    packed: np.ndarray,
    context: NormalizationContext,
    *,
    reference_pixels: np.ndarray | None = None,
) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[-1] != 4:
        raise ValueError("packed Bayer 必须是 HxWx4")
    if not np.all(np.isfinite(packed)):
        raise ValueError("降噪结果包含 NaN 或无穷值")
    height, width, _ = packed.shape
    output = np.empty((height * 2, width * 2), dtype=context.dtype)
    limits = np.iinfo(context.dtype)
    if reference_pixels is not None and (
        reference_pixels.shape != output.shape or not np.issubdtype(reference_pixels.dtype, np.integer)
    ):
        raise ValueError("参考 Bayer 必须是与输出同尺寸的整数数组")
    white = context.black_by_position + context.range_by_position
    for position, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        plane = np.array(packed[:, :, position], dtype=np.float32, copy=True)
        np.clip(plane, 0.0, 1.0, out=plane)
        np.multiply(plane, context.range_by_position[position], out=plane)
        np.add(plane, context.black_by_position[position], out=plane)
        np.clip(plane, limits.min, limits.max, out=plane)
        np.rint(plane, out=plane)
        output_plane = output[row::2, column::2]
        output_plane[:] = plane.astype(context.dtype, copy=False)
        if reference_pixels is not None:
            source_plane = reference_pixels[row::2, column::2]
            protected = (source_plane < context.black_by_position[position]) | (source_plane > white[position])
            output_plane[protected] = source_plane[protected].astype(context.dtype, copy=False)
    return output


def tiled_inference(
    image: np.ndarray,
    infer: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 1024,
    overlap: int = 64,
    on_progress: Callable[[int, int], None] | None = None,
    cancellation: CancellationToken | None = None,
    output_buffer: np.ndarray | None = None,
    weights_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Run HWC inference with overlap-add blending and no hard seams."""
    if image.ndim != 3:
        raise ValueError("image 必须是 HxWxC")
    if tile_size <= overlap * 2:
        raise ValueError("tile_size 必须大于两倍 overlap")
    if cancellation is not None:
        cancellation.check()
    height, width, channels = image.shape
    if height <= tile_size and width <= tile_size:
        result = infer(image)
        if cancellation is not None:
            cancellation.check()
        if result.shape != image.shape:
            raise ValueError("模型输出尺寸与输入不一致")
        if on_progress:
            on_progress(1, 1)
        return result

    step = tile_size - overlap
    if output_buffer is None:
        output = np.zeros_like(image, dtype=np.float32)
    else:
        if output_buffer.shape != image.shape or output_buffer.dtype != np.float32:
            raise ValueError("输出缓冲必须是与输入同尺寸的 float32 数组")
        if np.shares_memory(output_buffer, image):
            raise ValueError("输出缓冲不能与输入共享内存")
        output = output_buffer
        output.fill(0)
    if weights_buffer is None:
        weights = np.zeros((height, width), dtype=np.float32)
    else:
        if weights_buffer.shape != (height, width) or weights_buffer.dtype != np.float32:
            raise ValueError("权重缓冲必须是 HxW float32 数组")
        if np.shares_memory(weights_buffer, image) or np.shares_memory(weights_buffer, output):
            raise ValueError("权重缓冲不能与其他缓冲共享内存")
        weights = weights_buffer
        weights.fill(0)
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
            if cancellation is not None:
                cancellation.check()
            right = min(width, left + tile_size)
            tile = image[top:bottom, left:right]
            prediction = infer(tile)
            if cancellation is not None:
                cancellation.check()
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
            weights[top:bottom, left:right] += window
            completed += 1
            if on_progress:
                on_progress(completed, total)
    np.maximum(weights, 1e-8, out=weights)
    np.divide(output, weights[:, :, None], out=output)
    return output
