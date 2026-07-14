from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from arw_denoise.processor import AutoProcessingSettings, SmartRawProcessor
from arw_denoise.task_control import ProcessingCancelled, ProgressEvent, ProgressTracker, TaskController


def verify(source: Path, output_dir: Path, stage: str) -> dict[str, object]:
    output = output_dir / f"cancel-{stage}.dng"
    output.unlink(missing_ok=True)
    controller: TaskController
    requested_at: float | None = None
    timer: threading.Timer | None = None

    def progress(event: ProgressEvent) -> None:
        nonlocal requested_at, timer
        should_cancel = (
            stage == "tile" and event.phase == "denoising" and event.phase_progress > 0
        ) or (stage == "writing" and event.phase == "writing" and event.phase_progress == 0)
        if should_cancel and requested_at is None:
            requested_at = time.perf_counter()
            if stage == "writing":
                timer = threading.Timer(0.05, controller.cancel)
                timer.start()
            else:
                controller.cancel()

    controller = TaskController(progress_tracker=ProgressTracker(job_id=None, on_progress=progress))
    started = time.perf_counter()
    try:
        SmartRawProcessor().process(
            source,
            output,
            AutoProcessingSettings(mode="gpu"),
            control=controller,
        )
    except ProcessingCancelled:
        cancelled_at = time.perf_counter()
    else:
        raise RuntimeError(f"{stage} cancellation did not stop processing")
    finally:
        if timer is not None:
            timer.join(timeout=2)
    if requested_at is None:
        raise RuntimeError(f"{stage} cancellation checkpoint was not reached")
    leftovers = list(output_dir.glob(f".{output.stem}.*.processing.dng"))
    if output.exists() or leftovers:
        raise RuntimeError(f"{stage} cancellation left published or temporary DNG files")
    return {
        "stage": stage,
        "total_seconds": cancelled_at - started,
        "cancel_latency_seconds": cancelled_at - requested_at,
        "output_absent": True,
        "temporary_absent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real GPU tile and dnglab cancellation")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Missing ARW: {args.source}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [verify(args.source, args.output_dir, stage) for stage in ("tile", "writing")]
    text = json.dumps({"schema_version": 1, "results": results}, ensure_ascii=False, indent=2) + "\n"
    if args.json:
        args.json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
