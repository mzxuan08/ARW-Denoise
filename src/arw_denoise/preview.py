from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PREVIEW_RENDER_VERSION = "srgb-v1"


@dataclass(frozen=True)
class PreviewPair:
    source: np.ndarray
    denoised: np.ndarray

    def validate(self) -> None:
        if self.source.dtype != np.uint8 or self.denoised.dtype != np.uint8:
            raise ValueError("预览必须是 8-bit RGB")
        if self.source.ndim != 3 or self.source.shape[-1] != 3:
            raise ValueError("源图预览必须是 HxWx3")
        if self.denoised.shape != self.source.shape:
            raise ValueError("对比预览的方向和尺寸不一致")


def _rawpy_module():
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("缺少 rawpy，无法生成 RAW 预览") from exc
    return rawpy


def _read_white_balance(path: Path, *, rawpy_module) -> tuple[float, float, float, float]:
    with rawpy_module.imread(str(path)) as raw:
        values = tuple(float(value) for value in raw.camera_whitebalance)
    if len(values) != 4 or not all(np.isfinite(values)) or max(values) <= 0:
        raise RuntimeError("源 RAW 缺少可用的相机白平衡")
    return values  # type: ignore[return-value]


def render_srgb(
    path: Path,
    *,
    white_balance: tuple[float, float, float, float],
    half_size: bool,
    rawpy_module=None,
) -> np.ndarray:
    rawpy_module = rawpy_module or _rawpy_module()
    with rawpy_module.imread(str(path)) as raw:
        image = raw.postprocess(
            user_wb=list(white_balance),
            use_camera_wb=False,
            use_auto_wb=False,
            no_auto_bright=True,
            bright=1.0,
            gamma=(2.222, 4.5),
            output_color=rawpy_module.ColorSpace.sRGB,
            output_bps=8,
            half_size=half_size,
        )
    result = np.ascontiguousarray(image, dtype=np.uint8)
    if result.ndim != 3 or result.shape[-1] != 3:
        raise RuntimeError("显影器未返回 RGB 预览")
    return result


def render_preview_pair(
    source: Path,
    denoised: Path,
    *,
    half_size: bool = True,
    rawpy_module=None,
    renderer: Callable[..., np.ndarray] = render_srgb,
) -> PreviewPair:
    source = Path(source)
    denoised = Path(denoised)
    if not source.is_file() or not denoised.is_file():
        raise FileNotFoundError("源 ARW 或降噪 DNG 不存在")
    rawpy_module = rawpy_module or _rawpy_module()
    white_balance = _read_white_balance(source, rawpy_module=rawpy_module)
    original = renderer(
        source, white_balance=white_balance, half_size=half_size, rawpy_module=rawpy_module
    )
    processed = renderer(
        denoised, white_balance=white_balance, half_size=half_size, rawpy_module=rawpy_module
    )
    source_orientation = original.shape[0] >= original.shape[1]
    output_orientation = processed.shape[0] >= processed.shape[1]
    if source_orientation != output_orientation:
        raise ValueError("对比预览的旋转方向不一致")
    if original.shape != processed.shape:
        height = min(original.shape[0], processed.shape[0])
        width = min(original.shape[1], processed.shape[1])

        def center_crop(value: np.ndarray) -> np.ndarray:
            top = (value.shape[0] - height) // 2
            left = (value.shape[1] - width) // 2
            return np.ascontiguousarray(value[top:top + height, left:left + width])

        original = center_crop(original)
        processed = center_crop(processed)
    pair = PreviewPair(original, processed)
    pair.validate()
    return pair
