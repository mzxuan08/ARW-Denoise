from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import AppPaths
from .dnglab import DngLabClient
from .domain import ArwDenoiseError
from .gui import run_gui
from .jobs import JobStore
from .processor import CpuRawProcessor, ProcessingSettings
from .raw import RawPyDecoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arw-denoise", description="Sony ARW Bayer RAW 批量降噪")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="启动桌面界面")
    probe = sub.add_parser("probe", help="读取 RAW 元数据但不处理像素")
    probe.add_argument("source", type=Path)
    convert = sub.add_parser("dng-convert", help="建立未经降噪的 dnglab 兼容 DNG")
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--dnglab", type=Path)
    process = sub.add_parser("process-cpu", help="执行 CPU Bayer 降噪并写入 CFA DNG")
    process.add_argument("source", type=Path)
    process.add_argument("output", type=Path)
    process.add_argument("--dnglab", type=Path)
    process.add_argument("--strength", type=float, default=1.0)
    queue = sub.add_parser("queue-add", help="把 ARW 加入持久化队列")
    queue.add_argument("source", type=Path)
    queue.add_argument("--output-dir", type=Path)
    sub.add_parser("queue-list", help="列出持久化队列")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "gui"):
            return run_gui()
        if args.command == "probe":
            data = asdict(RawPyDecoder().probe(args.source))
            data["path"] = str(data["path"])
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0
        if args.command == "dng-convert":
            result = DngLabClient(args.dnglab).compatibility_convert(args.source, args.output)
            print(json.dumps({"output": str(result.output), "dnglab": result.version}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "process-cpu":
            client = DngLabClient(args.dnglab)
            result = CpuRawProcessor(dnglab=client).process(
                args.source, args.output, ProcessingSettings(strength=args.strength)
            )
            print(json.dumps({"output": str(result.output), "dnglab": result.version, "mode": "cpu"}, ensure_ascii=False, indent=2))
            return 0
        paths = AppPaths.default()
        paths.ensure()
        store = JobStore(paths.database)
        if args.command == "queue-add":
            output_dir = args.output_dir or args.source.parent / "DNG_Denoised"
            job = store.add_with_available_output(args.source, output_dir)
            print(f"已加入任务 #{job.id}: {job.source_path.name}")
            return 0
        if args.command == "queue-list":
            for job in store.list():
                print(f"{job.id}\t{job.state}\t{job.source_path.name}\t{job.output_path}")
            return 0
        parser.error("未知命令")
    except (ArwDenoiseError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 0
