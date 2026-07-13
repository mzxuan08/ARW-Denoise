from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


SETTINGS_SCHEMA_VERSION = 1
OUTPUT_STRATEGIES = {"source_subfolder", "fixed"}
ENGINE_MODES = {"auto", "gpu", "cpu"}


@dataclass(frozen=True)
class AppSettings:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    default_import_dir: str | None = None
    default_output_dir: str | None = None
    output_strategy: str = "source_subfolder"
    engine_mode: str = "auto"
    advanced_expanded: bool = False
    strength: float | None = None
    chroma_noise: float | None = None
    detail_protection: float | None = None
    artifact_suppression: float | None = None

    def validate(self) -> None:
        if self.schema_version != SETTINGS_SCHEMA_VERSION:
            raise ValueError("不支持的设置版本")
        if self.output_strategy not in OUTPUT_STRATEGIES:
            raise ValueError("未知导出策略")
        if self.engine_mode not in ENGINE_MODES:
            raise ValueError("未知处理引擎")
        for name, value, maximum in (
            ("strength", self.strength, 2.0),
            ("chroma_noise", self.chroma_noise, 1.0),
            ("detail_protection", self.detail_protection, 1.0),
            ("artifact_suppression", self.artifact_suppression, 1.0),
        ):
            if value is not None and not 0.0 <= value <= maximum:
                raise ValueError(f"{name} 超出范围")
        if self.output_strategy == "fixed" and not self.default_output_dir:
            raise ValueError("固定导出目录模式需要设置导出目录")

    @property
    def import_path(self) -> Path | None:
        return Path(self.default_import_dir) if self.default_import_dir else None

    @property
    def output_path(self) -> Path | None:
        return Path(self.default_output_dir) if self.default_output_dir else None


def resolve_output_dir(source: Path, settings: AppSettings) -> Path:
    settings.validate()
    if settings.output_strategy == "fixed":
        assert settings.default_output_dir is not None
        return Path(settings.default_output_dir).expanduser().resolve()
    return Path(source).expanduser().resolve().parent / "DNG_Denoised"


class SettingsStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("设置根节点必须是对象")
            known = set(AppSettings.__dataclass_fields__)
            settings = AppSettings(**{key: value for key, value in document.items() if key in known})
            settings.validate()
            return settings
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        settings.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
