from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .compatibility import discover_arw_files, scan_arw_files, write_compatibility_report
from .config import AppPaths
from .dnglab import DngLabClient
from .domain import ArwDenoiseError
from .gui import run_gui
from .gpu_probe import create_default_gpu_probe
from .jobs import JobStore
from .processor import AutoProcessingSettings, CpuRawProcessor, ProcessingSettings, SmartRawProcessor
from .raw import RawPyDecoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arw-denoise", description="Sony ARW Bayer RAW 批量降噪")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", help="启动桌面界面")
    probe = sub.add_parser("probe", help="读取 RAW 元数据但不处理像素")
    probe.add_argument("source", type=Path)
    scan = sub.add_parser("scan", help="批量预检 Sony ARW 兼容性")
    scan.add_argument("sources", type=Path, nargs="+")
    scan.add_argument("--workers", type=int, default=2)
    scan.add_argument("--report", type=Path)
    scan.add_argument("--no-recursive", action="store_true")
    convert = sub.add_parser("dng-convert", help="建立未经降噪的 dnglab 兼容 DNG")
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--dnglab", type=Path)
    process = sub.add_parser("process-cpu", help="执行 CPU Bayer 降噪并写入 CFA DNG")
    process.add_argument("source", type=Path)
    process.add_argument("output", type=Path)
    process.add_argument("--dnglab", type=Path)
    process.add_argument("--strength", type=float, default=1.0)
    smart = sub.add_parser("process", help="自动选择 GPU/CPU 执行 Bayer 降噪并写入 CFA DNG")
    smart.add_argument("source", type=Path)
    smart.add_argument("output", type=Path)
    smart.add_argument("--mode", choices=("auto", "gpu", "cpu"), default="auto")
    smart.add_argument("--strength", type=float)
    smart.add_argument("--chroma-noise", type=float)
    smart.add_argument("--detail-protection", type=float)
    smart.add_argument("--artifact-suppression", type=float)
    sub.add_parser("gpu-probe", help="执行一次真实 PMRID CUDA 推理自检")
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
        if args.command == "scan":
            paths: list[Path] = []
            seen: set[str] = set()
            for source in args.sources:
                candidates = (
                    discover_arw_files(source, recursive=not args.no_recursive)
                    if source.is_dir()
                    else [source.expanduser().resolve()]
                )
                for candidate in candidates:
                    key = str(candidate).casefold()
                    if key not in seen:
                        seen.add(key)
                        paths.append(candidate)
            if not paths:
                raise ValueError("没有找到可预检的 ARW 文件")
            results = scan_arw_files(paths, max_workers=args.workers)
            supported = sum(result.supported for result in results)
            report = write_compatibility_report(results, args.report) if args.report else None
            print(
                json.dumps(
                    {
                        "total": len(results),
                        "supported": supported,
                        "unsupported": len(results) - supported,
                        "report": str(report) if report else None,
                        "files": [result.to_document() for result in results],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if supported == len(results) else 4
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
        if args.command == "gpu-probe":
            result = create_default_gpu_probe().run(force=True)
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
            return 0 if result.success else 3
        if args.command == "process":
            result = SmartRawProcessor().process(
                args.source,
                args.output,
                AutoProcessingSettings(
                    mode=args.mode,
                    strength=args.strength,
                    chroma_noise=args.chroma_noise,
                    detail_protection=args.detail_protection,
                    artifact_suppression=args.artifact_suppression,
                ),
            )
            print(
                json.dumps(
                    {
                        "output": str(result.dng.output),
                        "dnglab": result.dng.version,
                        "engine": asdict(result.engine),
                        "stats": asdict(result.stats),
                        "automatic": asdict(result.automatic),
                        "postprocess": asdict(result.postprocess),
                        "fallback_reason": result.fallback_reason,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
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
