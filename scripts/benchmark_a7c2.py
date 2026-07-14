from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from arw_denoise.metrics import BenchmarkSample, ResourceMonitor, benchmark_markdown
from arw_denoise.processor import AutoProcessingSettings, SmartRawProcessor
from arw_denoise.task_control import ProgressEvent, ProgressTracker, TaskController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark A7C II ARW processing without modifying sources")
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mode", choices=("auto", "gpu", "cpu"), default="gpu")
    parser.add_argument("--prefetch", action="store_true", help="Decode the next RAW on one CPU thread")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def run_sample(
    processor: SmartRawProcessor,
    source: Path,
    output: Path,
    *,
    run: int,
    warmup: bool,
    mode: str,
    decoded_frame=None,
) -> BenchmarkSample:
    phase_bounds: dict[str, list[float]] = {}

    def progress(event: ProgressEvent) -> None:
        phase_bounds.setdefault(event.phase, []).append(event.timestamp)

    tracker = ProgressTracker(job_id=None, on_progress=progress)
    monitor = ResourceMonitor().start()
    started = time.perf_counter()
    result = None
    error = None
    try:
        result = processor.process(
            source,
            output,
            AutoProcessingSettings(mode=mode),
            control=TaskController(progress_tracker=tracker),
            decoded_frame=decoded_frame,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - started
    peaks = monitor.stop()
    stages = {
        phase: max(0.0, values[-1] - values[0])
        for phase, values in phase_bounds.items()
        if len(values) >= 2
    }
    if peaks.sampling_error:
        stages["sampling_warning"] = 0.0
    return BenchmarkSample(
        source=source.name,
        run=run,
        warmup=warmup,
        total_seconds=total,
        stage_seconds=stages,
        peak_ram_mb=peaks.ram_mb,
        peak_vram_mb=peaks.vram_mb,
        engine=result.engine.engine_id if result else None,
        provider=result.engine.provider if result else None,
        error=error,
    )


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("--runs must be >= 1 and --warmups must be >= 0")
    sources = [path.resolve() for path in args.sources]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing source: {', '.join(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = SmartRawProcessor()
    samples: list[BenchmarkSample] = []
    tasks: list[tuple[Path, bool, int]] = []
    for warmup_index in range(args.warmups):
        tasks.extend((source, True, warmup_index + 1) for source in sources)
    for run_index in range(args.runs):
        tasks.extend((source, False, run_index + 1) for source in sources)
    executor = ThreadPoolExecutor(max_workers=1) if args.prefetch else None
    prefetched: Future | None = (
        executor.submit(processor.decoder.decode, tasks[0][0]) if executor is not None and tasks else None
    )
    measured_started: float | None = None
    try:
        for index, (source, warmup, number) in enumerate(tasks):
            if not warmup and measured_started is None:
                measured_started = time.perf_counter()
            current = prefetched
            prefetched = None
            decoded_frame = current.result() if current is not None else None
            if executor is not None and index + 1 < len(tasks):
                prefetched = executor.submit(processor.decoder.decode, tasks[index + 1][0])
            label = "warmup" if warmup else "run"
            output = args.output_dir / f"{source.stem}_{label}_{number}_DN.dng"
            sample = run_sample(
                processor,
                source,
                output,
                run=number,
                warmup=warmup,
                mode=args.mode,
                decoded_frame=decoded_frame,
            )
            samples.append(sample)
            print(json.dumps(sample.to_dict(), ensure_ascii=False), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    measured_wall = time.perf_counter() - measured_started if measured_started is not None else None

    measured = [sample.total_seconds for sample in samples if not sample.warmup and not sample.error]
    payload = {
        "schema_version": 1,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mode": args.mode,
            "warmups": args.warmups,
            "runs": args.runs,
            "prefetch": args.prefetch,
        },
        "summary": {
            "successful_runs": len(measured),
            "median_total_seconds": statistics.median(measured) if measured else None,
            "measured_wall_seconds": measured_wall,
            "throughput_seconds_per_file": measured_wall / len(measured) if measured and measured_wall else None,
        },
        "samples": [sample.to_dict() for sample in samples],
    }
    json_path = args.json or args.output_dir / "benchmark.json"
    markdown_path = args.markdown or args.output_dir / "benchmark.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(benchmark_markdown(samples), encoding="utf-8")
    return 0 if measured else 1


if __name__ == "__main__":
    raise SystemExit(main())
