from __future__ import annotations

import json
from pathlib import Path

import pytest

from arw_denoise.settings import AppSettings, SettingsStore, resolve_output_dir


def test_missing_or_corrupt_settings_use_safe_automatic_defaults(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    assert store.load() == AppSettings()
    store.path.write_text("not-json", encoding="utf-8")
    assert store.load() == AppSettings()


def test_settings_round_trip_chinese_paths_and_advanced_values(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    settings = AppSettings(
        default_import_dir=str(tmp_path / "导入 照片"),
        default_output_dir=str(tmp_path / "导出 DNG"),
        output_strategy="fixed",
        engine_mode="gpu",
        advanced_expanded=True,
        advanced_enabled=True,
        strength=0.8,
        chroma_noise=0.6,
        detail_protection=0.9,
        artifact_suppression=0.7,
    )
    store.save(settings)
    assert store.load() == settings
    assert not store.path.with_suffix(".json.tmp").exists()


def test_old_settings_ignore_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"engine_mode": "cpu", "retired_option": 42}), encoding="utf-8")
    assert SettingsStore(path).load().engine_mode == "cpu"


def test_output_directory_policy_is_resolved_per_source(tmp_path: Path) -> None:
    source = tmp_path / "card" / "a.ARW"
    assert resolve_output_dir(source, AppSettings()) == source.parent.resolve() / "DNG_Denoised"
    fixed = tmp_path / "成品 DNG"
    configured = AppSettings(default_output_dir=str(fixed), output_strategy="fixed")
    assert resolve_output_dir(source, configured) == fixed.resolve()


def test_fixed_strategy_requires_a_directory() -> None:
    with pytest.raises(ValueError, match="导出目录"):
        AppSettings(output_strategy="fixed").validate()
