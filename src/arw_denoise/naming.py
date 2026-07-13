from __future__ import annotations

from pathlib import Path


def available_output_path(source: Path, output_dir: Path, suffix: str = "_DN") -> Path:
    output_dir = Path(output_dir)
    candidate = output_dir / f"{Path(source).stem}{suffix}.dng"
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        candidate = output_dir / f"{Path(source).stem}{suffix}_{index}.dng"
        if not candidate.exists():
            return candidate
        index += 1

