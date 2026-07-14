from __future__ import annotations

from pathlib import Path

import pytest

from arw_denoise.gui_helpers import (
    explorer_arguments,
    can_preview,
    format_duration,
    job_parameters,
    open_in_explorer,
    progress_eta,
    queue_progress,
    source_summary,
)
from arw_denoise.settings import AppSettings


def test_default_automatic_mode_does_not_snapshot_manual_overrides() -> None:
    settings = AppSettings(strength=1.2, chroma_noise=0.8, advanced_enabled=False)
    assert job_parameters(settings) == {}


def test_enabled_advanced_values_are_snapshotted_for_new_jobs() -> None:
    settings = AppSettings(
        advanced_enabled=True,
        strength=0.8,
        chroma_noise=0.6,
        detail_protection=0.9,
        artifact_suppression=0.7,
    )
    assert job_parameters(settings) == {
        "strength": 0.8,
        "chroma_noise": 0.6,
        "detail_protection": 0.9,
        "artifact_suppression": 0.7,
    }


def test_explorer_uses_argument_list_for_chinese_path_with_spaces(tmp_path: Path) -> None:
    folder = tmp_path / "导出 DNG"
    folder.mkdir()
    calls = []
    open_in_explorer(folder, launcher=lambda args, **kwargs: calls.append((args, kwargs)))
    assert calls == [(["explorer.exe", str(folder.resolve())], {"shell": False})]


def test_locate_file_uses_select_argument(tmp_path: Path) -> None:
    output = tmp_path / "成品 DNG.dng"
    output.write_bytes(b"dng")
    args = explorer_arguments(output, select_file=True)
    assert args == ["explorer.exe", f"/select,{output.resolve()}"]


def test_open_missing_output_reports_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="找不到"):
        open_in_explorer(tmp_path / "missing")


def test_progress_display_helpers_do_not_show_false_precision() -> None:
    class Item:
        def __init__(self, state: str, overall_progress: float = 0.0):
            self.state = state
            self.overall_progress = overall_progress

    jobs = [Item("completed"), Item("denoising", 0.5), Item("queued")]
    assert queue_progress(jobs) == pytest.approx(0.5)
    assert progress_eta(4.0, 0.05) is None
    assert progress_eta(4.0, 0.5) == pytest.approx(4.0)
    assert format_duration(None) == "--"
    assert format_duration(65) == "01:05"


def test_preview_requires_completed_job_and_both_files(tmp_path: Path) -> None:
    source, output = tmp_path / "a.ARW", tmp_path / "a.dng"
    source.write_bytes(b"raw")
    output.write_bytes(b"dng")

    class Job:
        state = "completed"
        source_path = source
        output_path = output

    assert can_preview(Job())
    Job.state = "failed"
    assert not can_preview(Job())


def test_source_summary_formats_preflight_camera_metadata() -> None:
    parameters = {
        "_source": {
            "make": "Sony",
            "model": "ILCE-7CM2",
            "iso": 400,
            "width": 7032,
            "height": 4688,
            "bits_per_sample": 16,
        }
    }
    assert source_summary(parameters) == "Sony ILCE-7CM2 · ISO 400 · 7032×4688 · 16 bit"
    assert source_summary({}) == ""
