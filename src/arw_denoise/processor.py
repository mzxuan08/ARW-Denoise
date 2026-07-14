from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .auto_tune import AutoDenoiseConfig, tune_automatic
from .denoise import HaarWaveletDenoiser
from .dnglab import DngLabClient, DngLabResult
from .domain import RawFrame
from .engines import CpuHaarEngine, DenoiseRequest, DenoiseResult, EngineInfo, EngineRunStats
from .gpu_probe import query_nvidia_device
from .model_manifest import default_model_root, load_manifest
from .onnx_engine import GpuRuntimeError, OnnxRuntimeEngine
from .pipeline import pack_normalized_bayer, tiled_inference, unpack_normalized_bayer
from .postprocess import PostprocessSettings, postprocess_raw
from .raw import RawDecoder, RawPyDecoder
from .tile_scheduler import AdaptiveTileRunner
from .task_control import TaskController


@dataclass(frozen=True)
class ProcessingSettings:
    strength: float = 1.0
    tile_size: int = 1024
    overlap: int = 64


@dataclass(frozen=True)
class AutoProcessingSettings:
    mode: str = "auto"
    strength: float | None = None
    chroma_noise: float | None = None
    detail_protection: float | None = None
    artifact_suppression: float | None = None
    cpu_tile_size: int = 1024
    cpu_overlap: int = 64

    def validate(self) -> None:
        if self.mode not in {"auto", "gpu", "cpu"}:
            raise ValueError("处理模式必须是 auto、gpu 或 cpu")
        if self.cpu_tile_size <= 2 * self.cpu_overlap:
            raise ValueError("CPU 分块必须大于两倍重叠")
        values = (
            ("strength", self.strength, 0.0, 2.0),
            ("chroma_noise", self.chroma_noise, 0.0, 1.0),
            ("detail_protection", self.detail_protection, 0.0, 1.0),
            ("artifact_suppression", self.artifact_suppression, 0.0, 1.0),
        )
        for name, value, low, high in values:
            if value is not None and not low <= value <= high:
                raise ValueError(f"{name} 超出范围")


@dataclass(frozen=True)
class SmartProcessingResult:
    dng: DngLabResult
    engine: EngineInfo
    stats: EngineRunStats
    automatic: AutoDenoiseConfig
    postprocess: PostprocessSettings
    fallback_reason: str | None = None


class CpuRawProcessor:
    def __init__(self, decoder: RawDecoder | None = None, dnglab: DngLabClient | None = None):
        self.decoder = decoder or RawPyDecoder()
        self.dnglab = dnglab or DngLabClient()
        self.denoiser = HaarWaveletDenoiser()

    def process(
        self,
        source: Path,
        output: Path,
        settings: ProcessingSettings | None = None,
        on_phase: Callable[[str], None] | None = None,
        control: TaskController | None = None,
    ) -> DngLabResult:
        control = control or TaskController()
        control.progress("decoding", 0, 1)
        settings = settings or ProcessingSettings()
        frame = self.decoder.decode(source)
        control.progress("decoding", 1, 1)
        if on_phase:
            on_phase("denoising")
        control.progress("denoising", 0, 1)
        packed, context = pack_normalized_bayer(frame.pixels, frame.metadata)
        processed = tiled_inference(
            packed,
            lambda tile: self.denoiser.denoise(tile, strength=settings.strength),
            tile_size=settings.tile_size,
            overlap=settings.overlap,
            cancellation=control.cancellation,
        )
        control.progress("denoising", 1, 1)
        control.progress("postprocessing", 0, 1)
        restored = unpack_normalized_bayer(processed, context, reference_pixels=frame.pixels)
        control.progress("postprocessing", 1, 1)
        if on_phase:
            on_phase("writing")
        control.progress("writing", 0, 1)
        result = self.dnglab.write_processed_cfa(
            source, output, restored, frame.metadata, cancellation=control.cancellation
        )
        control.progress("writing", 1, 1)
        control.progress("validating", 0, 1)
        control.progress("validating", 1, 1)
        return result


class SmartRawProcessor:
    def __init__(
        self,
        decoder: RawDecoder | None = None,
        dnglab: DngLabClient | None = None,
        *,
        gpu_runner: AdaptiveTileRunner | None = None,
        cpu_engine: CpuHaarEngine | None = None,
    ):
        self.decoder = decoder or RawPyDecoder()
        self.dnglab = dnglab or DngLabClient()
        self._gpu_runner = gpu_runner
        self._cpu_engine = cpu_engine or CpuHaarEngine()

    def _default_gpu_runner(self) -> AdaptiveTileRunner:
        root = default_model_root()
        manifest = load_manifest(root / "pmrid" / "manifest.json")
        cached_engine: list[OnnxRuntimeEngine | None] = [None]

        def engine_factory() -> OnnxRuntimeEngine:
            if cached_engine[0] is None:
                cached_engine[0] = OnnxRuntimeEngine(manifest, require_cuda=True)
            return cached_engine[0]

        def reset_engine() -> None:
            cached_engine[0] = None

        return AdaptiveTileRunner(
            engine_factory,
            memory_total_mb=query_nvidia_device().memory_total_mb,
            recommended_size=manifest.tiling.recommended_size,
            minimum_size=manifest.tiling.minimum_size,
            overlap=manifest.tiling.overlap,
            reset_engine=reset_engine,
        )

    def _resolve_postprocess(
        self,
        automatic: AutoDenoiseConfig,
        settings: AutoProcessingSettings,
    ) -> PostprocessSettings:
        return PostprocessSettings(
            strength=automatic.strength if settings.strength is None else settings.strength,
            chroma_noise=automatic.chroma_noise if settings.chroma_noise is None else settings.chroma_noise,
            detail_protection=(
                automatic.detail_protection
                if settings.detail_protection is None
                else settings.detail_protection
            ),
            artifact_suppression=(
                automatic.artifact_suppression
                if settings.artifact_suppression is None
                else settings.artifact_suppression
            ),
        )

    def _run_cpu(
        self,
        packed: np.ndarray,
        automatic: AutoDenoiseConfig,
        post: PostprocessSettings,
        settings: AutoProcessingSettings,
        control: TaskController,
        on_progress: Callable[[int, int], None] | None,
    ) -> DenoiseResult:
        seconds = 0.0

        def infer(tile: np.ndarray) -> np.ndarray:
            nonlocal seconds
            result = self._cpu_engine.run(
                DenoiseRequest(
                    packed=tile,
                    effective_iso=automatic.effective_iso,
                    strength=post.strength,
                )
            )
            seconds += result.stats.inference_seconds
            return result.packed

        output = tiled_inference(
            packed,
            infer,
            tile_size=settings.cpu_tile_size,
            overlap=settings.cpu_overlap,
            on_progress=on_progress,
            cancellation=control.cancellation,
        )
        return DenoiseResult(
            packed=output,
            engine=self._cpu_engine.info,
            stats=EngineRunStats(inference_seconds=seconds, tile_size=settings.cpu_tile_size),
        )

    def process(
        self,
        source: Path,
        output: Path,
        settings: AutoProcessingSettings | None = None,
        on_phase: Callable[[str], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        control: TaskController | None = None,
        decoded_frame: RawFrame | None = None,
    ) -> SmartProcessingResult:
        control = control or TaskController()
        control.progress("decoding", 0, 1)
        settings = settings or AutoProcessingSettings()
        settings.validate()
        frame = decoded_frame if decoded_frame is not None else self.decoder.decode(source)
        frame.validate()
        control.progress("decoding", 1, 1)
        packed, context = pack_normalized_bayer(frame.pixels, frame.metadata)
        automatic = tune_automatic(packed, frame.metadata)
        post = self._resolve_postprocess(automatic, settings)
        post.validate()
        if on_phase:
            on_phase("denoising")
        control.progress("denoising", 0, 1)

        best_tile_progress = 0.0

        def tile_progress(completed: int, total: int) -> None:
            nonlocal best_tile_progress
            if on_progress is not None:
                on_progress(completed, total)
            fraction = completed / total
            if fraction + 1e-12 >= best_tile_progress:
                best_tile_progress = fraction
                control.progress("denoising", completed, total)

        fallback_reason: str | None = None
        if settings.mode == "cpu":
            denoised = self._run_cpu(packed, automatic, post, settings, control, tile_progress)
        else:
            try:
                if self._gpu_runner is None:
                    self._gpu_runner = self._default_gpu_runner()
                model_result = self._gpu_runner.run(
                    DenoiseRequest(
                        packed=packed,
                        effective_iso=automatic.effective_iso,
                        strength=1.0,
                    ),
                    on_progress=tile_progress,
                    cancellation=control.cancellation,
                )
                control.progress("denoising", 1, 1)
                control.progress("postprocessing", 0, 1)
                processed = postprocess_raw(
                    packed,
                    model_result.packed,
                    post,
                    cancellation=control.cancellation,
                    out=model_result.packed,
                )
                denoised = DenoiseResult(
                    packed=processed,
                    engine=model_result.engine,
                    stats=model_result.stats,
                )
            except GpuRuntimeError as exc:
                if settings.mode == "gpu":
                    raise
                fallback_reason = str(exc)
                denoised = self._run_cpu(packed, automatic, post, settings, control, tile_progress)

        control.progress("denoising", 1, 1)
        control.progress("postprocessing", 1, 1)
        del packed
        restored = unpack_normalized_bayer(denoised.packed, context, reference_pixels=frame.pixels)
        if on_phase:
            on_phase("writing")
        control.progress("writing", 0, 1)
        dng_result = self.dnglab.write_processed_cfa(
            source,
            output,
            restored,
            frame.metadata,
            cancellation=control.cancellation,
        )
        control.progress("writing", 1, 1)
        control.progress("validating", 0, 1)
        control.progress("validating", 1, 1)
        return SmartProcessingResult(
            dng=dng_result,
            engine=denoised.engine,
            stats=denoised.stats,
            automatic=automatic,
            postprocess=post,
            fallback_reason=fallback_reason,
        )
