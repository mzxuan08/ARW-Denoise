from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "ArwDenoise"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    database: Path
    logs: Path
    settings: Path
    preview_cache: Path

    @classmethod
    def default(cls) -> "AppPaths":
        root = app_data_dir()
        return cls(
            root=root,
            database=root / "jobs.sqlite3",
            logs=root / "logs",
            settings=root / "settings.json",
            preview_cache=root / "preview-cache",
        )

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.preview_cache.mkdir(parents=True, exist_ok=True)
