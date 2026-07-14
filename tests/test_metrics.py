from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from arw_denoise.metrics import (
    BenchmarkSample,
    ResourceMonitor,
    StageTimer,
    benchmark_markdown,
    query_gpu_vram_mb,
    query_process_vram_mb,
)


def test_stage_timer_accumulates_repeat_stages() -> None:
    values = iter([1.0, 1.25, 2.0, 2.5])
    timer = StageTimer(clock=lambda: next(values))
    timer.start("decode")
    assert timer.stop("decode") == pytest.approx(0.25)
    timer.start("decode")
    timer.stop("decode")
    assert timer.seconds["decode"] == pytest.approx(0.75)


def test_vram_query_sums_only_current_process() -> None:
    def runner(_args, **_kwargs):
        return SimpleNamespace(stdout="42, 120.5\n7, 900\n42, 30\ninvalid\n")

    assert query_process_vram_mb(pid=42, runner=runner) == pytest.approx(150.5)


def test_gpu_vram_query_supports_wddm_fallback() -> None:
    def runner(_args, **_kwargs):
        return SimpleNamespace(stdout="812\ninvalid\n")

    assert query_gpu_vram_mb(runner=runner) == pytest.approx(812)


def test_monitor_records_peaks_and_reclaims_thread() -> None:
    rss = iter([10, 30, 20, 20])
    vram = iter([100.0, 250.0, 200.0, 200.0])

    class Process:
        def memory_info(self):
            return SimpleNamespace(rss=next(rss) * 1024 * 1024)

    monitor = ResourceMonitor(
        interval_seconds=0.005,
        process_factory=Process,
        vram_sampler=lambda: next(vram),
    ).start()
    time.sleep(0.025)
    peaks = monitor.stop()
    assert peaks.ram_mb == pytest.approx(30)
    assert peaks.vram_mb == pytest.approx(250)
    assert not any(thread.name == "resource-monitor" for thread in threading.enumerate())


def test_monitor_failure_does_not_escape_or_spin() -> None:
    monitor = ResourceMonitor(
        interval_seconds=0.005,
        process_factory=lambda: (_ for _ in ()).throw(RuntimeError("rss unavailable")),
        vram_sampler=lambda: (_ for _ in ()).throw(RuntimeError("smi unavailable")),
    ).start()
    time.sleep(0.015)
    peaks = monitor.stop()
    assert peaks.ram_mb is None
    assert peaks.vram_mb is None
    assert "rss unavailable" in (peaks.sampling_error or "")
    assert "smi unavailable" in (peaks.sampling_error or "")


def test_benchmark_markdown_is_stable_and_marks_missing_samples() -> None:
    sample = BenchmarkSample(
        source="a.ARW",
        run=1,
        warmup=False,
        total_seconds=2.5,
        stage_seconds={"denoising": 1.0},
        peak_ram_mb=100.0,
        peak_vram_mb=None,
        provider="CUDAExecutionProvider",
    )
    report = benchmark_markdown([sample])
    assert "| a.ARW | 1 | no | 2.500 | 100.0 | -- | CUDAExecutionProvider |" in report
