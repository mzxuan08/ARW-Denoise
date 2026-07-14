from pathlib import Path

from arw_denoise.cli import build_parser


def test_smart_process_cli_defaults_to_auto() -> None:
    args = build_parser().parse_args(["process", "in.ARW", "out.dng"])
    assert args.mode == "auto"
    assert args.source == Path("in.ARW")
    assert args.strength is None


def test_smart_process_cli_accepts_advanced_overrides() -> None:
    args = build_parser().parse_args(
        [
            "process",
            "in.ARW",
            "out.dng",
            "--mode",
            "gpu",
            "--strength",
            "0.8",
            "--detail-protection",
            "0.9",
        ]
    )
    assert args.mode == "gpu"
    assert args.strength == 0.8
    assert args.detail_protection == 0.9


def test_gpu_probe_cli_command() -> None:
    assert build_parser().parse_args(["gpu-probe"]).command == "gpu-probe"


def test_scan_cli_accepts_multiple_sources_and_report_options() -> None:
    args = build_parser().parse_args(
        ["scan", "card", "one.ARW", "--workers", "3", "--report", "report.json"]
    )
    assert args.command == "scan"
    assert args.sources == [Path("card"), Path("one.ARW")]
    assert args.workers == 3
    assert args.report == Path("report.json")

