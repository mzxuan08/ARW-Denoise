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

