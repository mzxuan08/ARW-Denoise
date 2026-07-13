from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from arw_denoise.jobs import JobStore


def run_stress(root: Path, count: int = 150) -> None:
    store = JobStore(root / "jobs.sqlite3")
    output = root / "output"
    jobs = [
        store.add_with_available_output(root / f"card-{index:03d}" / "same.ARW", output)
        for index in range(count)
    ]
    if len({job.output_path for job in jobs}) != count:
        raise RuntimeError("队列输出预留发生冲突")

    for job in jobs[:50]:
        for state in ("decoding", "denoising", "writing", "validating", "completed"):
            store.transition(job.id, state)
    for job in jobs[50:75]:
        store.transition(job.id, "decoding")
        store.transition(job.id, "failed", "injected failure")
    for job in jobs[75:100]:
        store.transition(job.id, "decoding")
        store.transition(job.id, "denoising")
    for job in jobs[100:125]:
        store.transition(job.id, "cancelled")

    if store.recover_interrupted() != 25:
        raise RuntimeError("中断任务恢复数量不正确")
    for job in store.list("failed") + store.list("cancelled"):
        store.transition(job.id, "queued")
    states = {job.state for job in store.list()}
    if states != {"queued", "completed"}:
        raise RuntimeError(f"压力测试后队列状态异常：{states}")
    if len(store.list()) != count:
        raise RuntimeError("队列项数发生变化")


def main() -> int:
    parser = argparse.ArgumentParser(description="ARW Denoise persistent queue stress test")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.count < 150:
        parser.error("count must be at least 150")
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run_stress(args.work_dir, args.count)
    else:
        with tempfile.TemporaryDirectory(prefix="arw-denoise-stress-") as temporary:
            run_stress(Path(temporary), args.count)
    print(f"Queue stress passed: {args.count} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
