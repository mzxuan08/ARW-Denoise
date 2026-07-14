from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .task_control import CancellationToken


@dataclass(frozen=True)
class PostprocessSettings:
    strength: float = 1.0
    chroma_noise: float = 0.5
    detail_protection: float = 0.7
    artifact_suppression: float = 0.5

    def validate(self) -> None:
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 2.0:
            raise ValueError("降噪强度必须在 0–2")
        for name, value in (
            ("彩色噪点", self.chroma_noise),
            ("细节保护", self.detail_protection),
            ("伪影抑制", self.artifact_suppression),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}必须在 0–1")


def _validate_images(original: np.ndarray, model_output: np.ndarray) -> None:
    if original.ndim != 3 or original.shape[-1] != 4:
        raise ValueError("原始 packed Bayer 必须是 HxWx4")
    if model_output.shape != original.shape:
        raise ValueError("模型输出尺寸与原始 Bayer 不一致")
    if not np.issubdtype(original.dtype, np.floating) or not np.issubdtype(model_output.dtype, np.floating):
        raise ValueError("RAW 后处理输入必须是浮点数组")
    if not np.isfinite(original).all() or not np.isfinite(model_output).all():
        raise ValueError("RAW 后处理输入包含 NaN 或无穷值")


def _edge_weight(original: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return np.ones(original.shape[:2], dtype=np.float32)
    luminance = original.mean(axis=2, dtype=np.float32)
    gradient = np.zeros_like(luminance)
    horizontal = np.abs(luminance[:, 1:] - luminance[:, :-1])
    vertical = np.abs(luminance[1:, :] - luminance[:-1, :])
    gradient[:, 1:] = np.maximum(gradient[:, 1:], horizontal)
    gradient[:, :-1] = np.maximum(gradient[:, :-1], horizontal)
    gradient[1:, :] = np.maximum(gradient[1:, :], vertical)
    gradient[:-1, :] = np.maximum(gradient[:-1, :], vertical)
    nonzero = gradient[gradient > 0]
    scale = float(np.quantile(nonzero, 0.90)) if nonzero.size else 1.0
    edge = np.clip(gradient / max(scale, 1e-5), 0.0, 1.0)
    return (1.0 - 0.82 * amount * edge).astype(np.float32)


def _control_chroma(delta: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return delta
    green = 0.5 * (delta[:, :, 1] + delta[:, :, 2])
    for channel, mix in ((0, amount), (3, amount), (1, 0.5 * amount), (2, 0.5 * amount)):
        delta[:, :, channel] *= 1.0 - mix
        delta[:, :, channel] += mix * green
    return delta


def _suppress_delta_spikes(delta: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return delta
    limit = 0.08 * (1.0 - amount) + 0.008
    for channel in range(delta.shape[2]):
        plane = delta[:, :, channel]
        padded = np.pad(plane, ((1, 1), (1, 1)), mode="reflect")
        neighbors = padded[:-2, 1:-1] + padded[2:, 1:-1]
        neighbors += padded[1:-1, :-2]
        neighbors += padded[1:-1, 2:]
        neighbors *= 0.25
        limited = plane - neighbors
        np.clip(limited, -limit, limit, out=limited)
        limited += neighbors
        plane *= 1.0 - amount
        plane += amount * limited
    return delta


def _preserve_channel_means(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    target = original.mean(axis=(0, 1), dtype=np.float64)
    corrected = result
    for _ in range(2):
        actual = corrected.mean(axis=(0, 1), dtype=np.float64)
        corrected += (target - actual).astype(np.float32)[None, None, :]
        np.clip(corrected, 0.0, 1.0, out=corrected)
    return corrected.astype(np.float32, copy=False)


def postprocess_raw(
    original: np.ndarray,
    model_output: np.ndarray,
    settings: PostprocessSettings,
    *,
    cancellation: CancellationToken | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    settings.validate()
    _validate_images(original, model_output)
    if cancellation is not None:
        cancellation.check()
    source = original.astype(np.float32, copy=False)
    prediction = model_output.astype(np.float32, copy=False)
    if np.array_equal(source, prediction) or settings.strength == 0.0:
        if out is None:
            return source.copy()
        if out.shape != source.shape or out.dtype != np.float32 or np.shares_memory(out, source):
            raise ValueError("后处理输出缓冲无效或与原始 RAW 共享内存")
        np.copyto(out, source)
        return out
    if out is None:
        delta = np.empty_like(source, dtype=np.float32)
    else:
        if out.shape != source.shape or out.dtype != np.float32 or np.shares_memory(out, source):
            raise ValueError("后处理输出缓冲无效或与原始 RAW 共享内存")
        delta = out
    np.subtract(prediction, source, out=delta)
    delta = _control_chroma(delta, settings.chroma_noise)
    if cancellation is not None:
        cancellation.check()
    delta = _suppress_delta_spikes(delta, settings.artifact_suppression)
    if cancellation is not None:
        cancellation.check()
    delta *= _edge_weight(source, settings.detail_protection)[:, :, None]
    if cancellation is not None:
        cancellation.check()
    delta *= settings.strength
    delta += source
    np.clip(delta, 0.0, 1.0, out=delta)
    result = delta
    result = _preserve_channel_means(result, source)
    if cancellation is not None:
        cancellation.check()
    if not np.isfinite(result).all():
        raise ValueError("RAW 后处理结果包含 NaN 或无穷值")
    return result

