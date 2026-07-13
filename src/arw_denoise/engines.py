from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .denoise import HaarWaveletDenoiser


@dataclass(frozen=True)
class DenoiseRequest:
    packed: np.ndarray
    effective_iso: float
    strength: float = 1.0

    def validate(self) -> None:
        if not isinstance(self.packed, np.ndarray) or self.packed.ndim != 3 or self.packed.shape[-1] != 4:
            raise ValueError("packed Bayer 必须是 HxWx4")
        if self.packed.shape[0] <= 0 or self.packed.shape[1] <= 0:
            raise ValueError("packed Bayer 尺寸无效")
        if not np.issubdtype(self.packed.dtype, np.floating):
            raise ValueError("packed Bayer 必须是浮点数组")
        if not np.isfinite(self.packed).all():
            raise ValueError("packed Bayer 包含 NaN 或无穷值")
        if not math.isfinite(self.effective_iso) or not 1.0 <= self.effective_iso <= 1_000_000.0:
            raise ValueError("effective_iso 无效")
        if not math.isfinite(self.strength) or not 0.0 <= self.strength <= 2.0:
            raise ValueError("strength 必须在 0–2")


@dataclass(frozen=True)
class EngineInfo:
    engine_id: str
    display_name: str
    provider: str
    is_gpu: bool
    model_id: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class EngineRunStats:
    inference_seconds: float
    tile_size: int | None = None
    peak_vram_mb: float | None = None


@dataclass(frozen=True)
class DenoiseResult:
    packed: np.ndarray
    engine: EngineInfo
    stats: EngineRunStats


class RawDenoiseEngine(Protocol):
    @property
    def info(self) -> EngineInfo: ...

    def run(self, request: DenoiseRequest) -> DenoiseResult: ...


class CpuHaarEngine:
    def __init__(self, denoiser: HaarWaveletDenoiser | None = None):
        self._denoiser = denoiser or HaarWaveletDenoiser()

    @property
    def info(self) -> EngineInfo:
        return EngineInfo(
            engine_id="cpu-haar",
            display_name="CPU Haar 保守降噪",
            provider="CPU",
            is_gpu=False,
        )

    def run(self, request: DenoiseRequest) -> DenoiseResult:
        request.validate()
        started = time.perf_counter()
        output = self._denoiser.denoise(request.packed.astype(np.float32, copy=False), request.strength)
        elapsed = time.perf_counter() - started
        if output.shape != request.packed.shape or not np.isfinite(output).all():
            raise RuntimeError("CPU 降噪引擎返回了无效结果")
        return DenoiseResult(
            packed=np.clip(output, 0.0, 1.0).astype(np.float32, copy=False),
            engine=self.info,
            stats=EngineRunStats(inference_seconds=elapsed),
        )

