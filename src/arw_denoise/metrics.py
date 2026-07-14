from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol


class MemoryProcess(Protocol):
    def memory_info(self) -> object: ...


@dataclass(frozen=True)
class ResourcePeaks:
    ram_mb: float | None
    vram_mb: float | None
    sampling_error: str | None = None


@dataclass
class StageTimer:
    clock: Callable[[], float] = time.perf_counter
    _started: dict[str, float] = field(default_factory=dict, init=False)
    seconds: dict[str, float] = field(default_factory=dict, init=False)

    def start(self, stage: str) -> None:
        if not stage or stage in self._started:
            raise ValueError("计时阶段为空或已经开始")
        self._started[stage] = self.clock()

    def stop(self, stage: str) -> float:
        try:
            started = self._started.pop(stage)
        except KeyError as exc:
            raise ValueError("计时阶段尚未开始") from exc
        elapsed = max(0.0, self.clock() - started)
        self.seconds[stage] = self.seconds.get(stage, 0.0) + elapsed
        return elapsed


def query_process_vram_mb(
    *,
    pid: int | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> float | None:
    target = os.getpid() if pid is None else pid
    result = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    total = 0.0
    matched = False
    for line in str(getattr(result, "stdout", "")).splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            row_pid, used = int(fields[0]), float(fields[1])
        except ValueError:
            continue
        if row_pid == target:
            total += used
            matched = True
    return total if matched else None


def query_gpu_vram_mb(
    *, runner: Callable[..., object] = subprocess.run
) -> float | None:
    """Return device-wide used VRAM when Windows WDDM hides per-process usage."""
    result = runner(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    values: list[float] = []
    for line in str(getattr(result, "stdout", "")).splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def query_vram_mb() -> float | None:
    process_value = query_process_vram_mb()
    return process_value if process_value is not None else query_gpu_vram_mb()


class ResourceMonitor:
    """Bounded background sampler; sampling failures never fail processing."""

    def __init__(
        self,
        *,
        interval_seconds: float = 0.2,
        process_factory: Callable[[], MemoryProcess] | None = None,
        vram_sampler: Callable[[], float | None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("采样间隔必须为正数")
        if process_factory is None:
            import psutil

            process_factory = psutil.Process
        self.interval_seconds = interval_seconds
        self._process_factory = process_factory
        self._vram_sampler = vram_sampler or query_vram_mb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_ram_mb: float | None = None
        self._peak_vram_mb: float | None = None
        self._errors: list[str] = []

    def start(self) -> "ResourceMonitor":
        if self._thread is not None:
            raise RuntimeError("资源监测已经启动")
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        try:
            process = self._process_factory()
        except Exception as exc:
            self._errors.append(f"RAM: {exc}")
            process = None
        while not self._stop.is_set():
            if process is not None:
                try:
                    rss = float(getattr(process.memory_info(), "rss")) / (1024 * 1024)
                    self._peak_ram_mb = rss if self._peak_ram_mb is None else max(self._peak_ram_mb, rss)
                except Exception as exc:
                    self._errors.append(f"RAM: {exc}")
                    process = None
            try:
                vram = self._vram_sampler()
                if vram is not None:
                    self._peak_vram_mb = vram if self._peak_vram_mb is None else max(self._peak_vram_mb, vram)
            except Exception as exc:
                self._errors.append(f"VRAM: {exc}")
                self._vram_sampler = lambda: None
            self._stop.wait(self.interval_seconds)

    def stop(self, *, timeout: float = 2.0) -> ResourcePeaks:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                self._errors.append("监测线程未在期限内退出")
            self._thread = None
        error = "; ".join(dict.fromkeys(self._errors)) or None
        return ResourcePeaks(self._peak_ram_mb, self._peak_vram_mb, error)

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.stop()


@dataclass(frozen=True)
class BenchmarkSample:
    source: str
    run: int
    warmup: bool
    total_seconds: float
    stage_seconds: dict[str, float]
    peak_ram_mb: float | None
    peak_vram_mb: float | None
    engine: str | None = None
    provider: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def benchmark_markdown(samples: list[BenchmarkSample]) -> str:
    lines = [
        "# ARW Denoise benchmark",
        "",
        "| Source | Run | Warmup | Total (s) | RAM (MB) | VRAM (MB) | Provider | Error |",
        "|---|---:|:---:|---:|---:|---:|---|---|",
    ]
    for sample in samples:
        ram = "--" if sample.peak_ram_mb is None else f"{sample.peak_ram_mb:.1f}"
        vram = "--" if sample.peak_vram_mb is None else f"{sample.peak_vram_mb:.1f}"
        lines.append(
            f"| {sample.source} | {sample.run} | {'yes' if sample.warmup else 'no'} | "
            f"{sample.total_seconds:.3f} | {ram} | {vram} | {sample.provider or '--'} | "
            f"{sample.error or ''} |"
        )
    return "\n".join(lines) + "\n"
