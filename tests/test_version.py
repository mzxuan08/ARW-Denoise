from __future__ import annotations

import re
from pathlib import Path

from arw_denoise import __version__


def test_package_version_matches_project_metadata() -> None:
    metadata = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', metadata, flags=re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)
