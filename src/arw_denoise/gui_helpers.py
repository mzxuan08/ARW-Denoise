from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .settings import AppSettings


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    rounded = int(round(seconds))
    minutes, remaining = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def queue_progress(jobs: Sequence[object]) -> float:
    if not jobs:
        return 0.0
    progress = 0.0
    for job in jobs:
        state = str(getattr(job, "state"))
        if state == "completed":
            progress += 1.0
        elif state not in {"queued", "failed", "cancelled"}:
            progress += float(getattr(job, "overall_progress", 0.0))
    return min(1.0, max(0.0, progress / len(jobs)))


def progress_eta(elapsed_seconds: float, overall_progress: float) -> float | None:
    """Estimate remaining time only after enough of the current file has run."""
    if elapsed_seconds <= 0 or overall_progress < 0.1 or overall_progress >= 1.0:
        return None
    return elapsed_seconds * (1.0 - overall_progress) / overall_progress


def can_preview(job: object | None) -> bool:
    if job is None or getattr(job, "state", None) != "completed":
        return False
    return Path(getattr(job, "source_path")).is_file() and Path(getattr(job, "output_path")).is_file()


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


def source_summary(parameters: dict) -> str:
    source = parameters.get("_source")
    if not isinstance(source, dict):
        return ""
    camera = " ".join(
        str(value).strip() for value in (source.get("make"), source.get("model")) if value
    )
    fields = [camera] if camera else []
    if source.get("iso") is not None:
        fields.append(f"ISO {source['iso']}")
    if source.get("width") and source.get("height"):
        fields.append(f"{source['width']}×{source['height']}")
    if source.get("bits_per_sample"):
        fields.append(f"{source['bits_per_sample']} bit")
    return " · ".join(fields)


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
