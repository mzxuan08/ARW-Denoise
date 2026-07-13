from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .settings import AppSettings


def job_parameters(settings: AppSettings) -> dict[str, float]:
    if not settings.advanced_enabled:
        return {}
    values = {
        "strength": settings.strength,
        "chroma_noise": settings.chroma_noise,
        "detail_protection": settings.detail_protection,
        "artifact_suppression": settings.artifact_suppression,
    }
    return {name: float(value) for name, value in values.items() if value is not None}


def explorer_arguments(path: Path, *, select_file: bool = False) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    if select_file:
        return ["explorer.exe", f"/select,{resolved}"]
    return ["explorer.exe", str(resolved)]


def open_in_explorer(
    path: Path,
    *,
    select_file: bool = False,
    launcher: Callable[..., object] = subprocess.Popen,
) -> None:
    resolved = Path(path).expanduser().resolve()
    if select_file:
        if not resolved.is_file():
            raise FileNotFoundError(f"找不到要定位的文件：{resolved}")
    elif not resolved.is_dir():
        raise FileNotFoundError(f"找不到要打开的目录：{resolved}")
    arguments: Sequence[str] = explorer_arguments(resolved, select_file=select_file)
    launcher(arguments, shell=False)
