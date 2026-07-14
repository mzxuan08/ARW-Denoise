from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arw_denoise.dnglab import DngLabResult
from arw_denoise.domain import RawFrame, RawMetadata
from arw_denoise.engines import DenoiseResult, EngineInfo, EngineRunStats
from arw_denoise.onnx_engine import GpuRuntimeError
from arw_denoise.processor import AutoProcessingSettings, SmartRawProcessor
from arw_denoise.task_control import ProcessingCancelled, ProgressTracker, TaskController


class FakeDecoder:
    def __init__(self, pixels: np.ndarray, metadata: RawMetadata):
        self.frame = RawFrame(metadata, pixels)

    def probe(self, path: Path) -> RawMetadata:
        return self.frame.metadata

    def decode(self, path: Path) -> RawFrame:
        return self.frame


class FakeWriter:
    def __init__(self):
        self.pixels: np.ndarray | None = None

    def write_processed_cfa(self, source, output, pixels, metadata, cancellation=None):
        self.pixels = pixels.copy()
        return DngLabResult(Path(output), "fake", {"valid": True})


class FakeGpuRunner:
    def __init__(self, output: np.ndarray | None = None, error: Exception | None = None):
        self.output = output
        self.error = error
        self.request = None

    def run(self, request, on_progress=None, cancellation=None):
        self.request = request
        if cancellation is not None:
            cancellation.check()
        if self.error:
            raise self.error
        if on_progress:
            on_progress(1, 1)
        output = self.output if self.output is not None else request.packed
        return DenoiseResult(
            packed=output.astype(np.float32),
            engine=EngineInfo(
                engine_id="onnx-pmrid",
                display_name="PMRID",
                provider="CUDAExecutionProvider",
                is_gpu=True,
                model_id="pmrid-general-raw",
                model_version="1.0.0",
            ),
            stats=EngineRunStats(inference_seconds=0.2, tile_size=512),
        )


def _fixture(tmp_path: Path) -> tuple[np.ndarray, RawMetadata]:
    rng = np.random.default_rng(41)
    pixels = np.clip(4500 + rng.normal(0, 250, (64, 64)), 512, 15360).astype(np.uint16)
    metadata = RawMetadata(
        path=tmp_path / "sample.ARW",
        width=64,
        height=64,
        raw_width=64,
        raw_height=64,
        cfa_pattern=(0, 1, 3, 2),
        color_description="RGBG",
        black_levels=(512, 512, 512, 512),
        white_level=15360,
        bits_per_sample=14,
        make="Sony",
        model="ILCE-7CM2",
        iso=3200,
        shutter_seconds=1 / 60,
    )
    return pixels, metadata


def test_smart_processor_uses_gpu_model_then_raw_postprocess(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)
    writer = FakeWriter()
    gpu = FakeGpuRunner()
    processor = SmartRawProcessor(
        decoder=FakeDecoder(pixels, metadata),
        dnglab=writer,
        gpu_runner=gpu,
    )
    phases: list[str] = []
    result = processor.process(metadata.path, tmp_path / "out.dng", on_phase=phases.append)

    assert result.engine.provider == "CUDAExecutionProvider"
    assert result.fallback_reason is None
    assert result.automatic.strategy_version == "pmrid-auto-v1"
    assert result.postprocess.strength == result.automatic.strength
    assert writer.pixels is not None
    assert writer.pixels.dtype == np.uint16
    assert writer.pixels.shape == pixels.shape
    assert phases == ["denoising", "writing"]
    assert gpu.request.strength == 1.0


def test_smart_processor_accepts_prefetched_frame_without_decoding_again(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)

    class MustNotDecode(FakeDecoder):
        def decode(self, path: Path) -> RawFrame:
            raise AssertionError("prefetched frame should be reused")

    processor = SmartRawProcessor(
        decoder=MustNotDecode(pixels, metadata),
        dnglab=FakeWriter(),
        gpu_runner=FakeGpuRunner(),
    )
    result = processor.process(
        metadata.path,
        tmp_path / "out.dng",
        decoded_frame=RawFrame(metadata, pixels),
    )
    assert result.engine.provider == "CUDAExecutionProvider"


def test_auto_mode_falls_back_to_cpu_and_records_reason(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)
    writer = FakeWriter()
    processor = SmartRawProcessor(
        decoder=FakeDecoder(pixels, metadata),
        dnglab=writer,
        gpu_runner=FakeGpuRunner(error=GpuRuntimeError("CUDA unavailable")),
    )
    result = processor.process(metadata.path, tmp_path / "out.dng")
    assert not result.engine.is_gpu
    assert "CUDA unavailable" in (result.fallback_reason or "")
    assert writer.pixels is not None


def test_explicit_gpu_mode_does_not_hide_failure(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)
    processor = SmartRawProcessor(
        decoder=FakeDecoder(pixels, metadata),
        dnglab=FakeWriter(),
        gpu_runner=FakeGpuRunner(error=GpuRuntimeError("CUDA unavailable")),
    )
    with pytest.raises(GpuRuntimeError, match="CUDA unavailable"):
        processor.process(
            metadata.path,
            tmp_path / "out.dng",
            AutoProcessingSettings(mode="gpu"),
        )


def test_advanced_overrides_replace_only_selected_auto_values(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)
    processor = SmartRawProcessor(
        decoder=FakeDecoder(pixels, metadata),
        dnglab=FakeWriter(),
        gpu_runner=FakeGpuRunner(),
    )
    result = processor.process(
        metadata.path,
        tmp_path / "out.dng",
        AutoProcessingSettings(strength=0.25, detail_protection=0.9),
    )
    assert result.postprocess.strength == 0.25
    assert result.postprocess.detail_protection == 0.9
    assert result.postprocess.chroma_noise == result.automatic.chroma_noise


def test_smart_processor_reports_all_weighted_phases(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)
    events = []
    control = TaskController(
        progress_tracker=ProgressTracker(job_id=3, on_progress=events.append)
    )
    SmartRawProcessor(
        decoder=FakeDecoder(pixels, metadata),
        dnglab=FakeWriter(),
        gpu_runner=FakeGpuRunner(),
    ).process(metadata.path, tmp_path / "out.dng", control=control)
    assert events[-1].overall == 1.0
    assert {event.phase for event in events} == {
        "decoding", "denoising", "postprocessing", "writing", "validating"
    }
    assert [event.overall for event in events] == sorted(event.overall for event in events)


def test_smart_processor_cancelled_during_gpu_does_not_write(tmp_path: Path) -> None:
    pixels, metadata = _fixture(tmp_path)
    writer = FakeWriter()
    control = TaskController()

    class CancellingRunner(FakeGpuRunner):
        def run(self, request, on_progress=None, cancellation=None):
            cancellation.cancel()
            cancellation.check()

    processor = SmartRawProcessor(
        decoder=FakeDecoder(pixels, metadata),
        dnglab=writer,
        gpu_runner=CancellingRunner(),
    )
    with pytest.raises(ProcessingCancelled):
        processor.process(metadata.path, tmp_path / "out.dng", control=control)
    assert writer.pixels is None

