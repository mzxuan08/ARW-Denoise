from __future__ import annotations

from pathlib import Path

import pytest

from arw_denoise.gui_helpers import explorer_arguments, job_parameters, open_in_explorer
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
