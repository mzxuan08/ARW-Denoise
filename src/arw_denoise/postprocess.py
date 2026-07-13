from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
    controlled = delta.copy()
    green = 0.5 * (delta[:, :, 1] + delta[:, :, 2])
    controlled[:, :, 0] = (1.0 - amount) * delta[:, :, 0] + amount * green
    controlled[:, :, 3] = (1.0 - amount) * delta[:, :, 3] + amount * green
    controlled[:, :, 1] = (1.0 - 0.5 * amount) * delta[:, :, 1] + 0.5 * amount * green
    controlled[:, :, 2] = (1.0 - 0.5 * amount) * delta[:, :, 2] + 0.5 * amount * green
    return controlled


def _suppress_delta_spikes(delta: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return delta
    padded = np.pad(delta, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    neighbors = 0.25 * (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    limit = 0.08 * (1.0 - amount) + 0.008
    limited = neighbors + np.clip(delta - neighbors, -limit, limit)
    return (1.0 - amount) * delta + amount * limited


def _preserve_channel_means(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    target = original.mean(axis=(0, 1), dtype=np.float64)
    corrected = result
    for _ in range(2):
        actual = corrected.mean(axis=(0, 1), dtype=np.float64)
        corrected = np.clip(corrected + (target - actual).astype(np.float32)[None, None, :], 0.0, 1.0)
    return corrected.astype(np.float32, copy=False)


def postprocess_raw(
    original: np.ndarray,
    model_output: np.ndarray,
    settings: PostprocessSettings,
) -> np.ndarray:
    settings.validate()
    _validate_images(original, model_output)
    source = original.astype(np.float32, copy=False)
    prediction = model_output.astype(np.float32, copy=False)
    if np.array_equal(source, prediction) or settings.strength == 0.0:
        return source.copy()
    delta = prediction - source
    delta = _control_chroma(delta, settings.chroma_noise)
    delta = _suppress_delta_spikes(delta, settings.artifact_suppression)
    delta *= _edge_weight(source, settings.detail_protection)[:, :, None]
    result = np.clip(source + settings.strength * delta, 0.0, 1.0)
    result = _preserve_channel_means(result, source)
    if not np.isfinite(result).all():
        raise ValueError("RAW 后处理结果包含 NaN 或无穷值")
    return result

