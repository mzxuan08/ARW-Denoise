from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_pmrid_model.ps1"


def _run_fetch(source: Path, destination: Path, expected: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-SourcePath",
            str(source),
            "-Destination",
            str(destination),
            "-ExpectedSha256",
            expected,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_upstream_record_pins_commit_and_checkpoint_hash() -> None:
    text = (ROOT / "vendor" / "pmrid" / "UPSTREAM.md").read_text(encoding="utf-8")
    assert "8ebb9e8e96559881dee957f34243933c5beb77dd" in text
    assert "9361614f3514d27351d81909f2215c0fdc38619c0288d936b7266485ac106c14" in text.lower()
    assert (ROOT / "vendor" / "pmrid" / "LICENSE").read_text(encoding="utf-8").startswith("Apache License")


def test_fetch_script_publishes_only_matching_asset(tmp_path: Path) -> None:
    source = tmp_path / "source.ckp"
    source.write_bytes(b"fixed model bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "nested" / "model.ckp"

    result = _run_fetch(source, destination, digest)

    assert result.returncode == 0, result.stderr
    assert destination.read_bytes() == source.read_bytes()


def test_fetch_script_rejects_wrong_hash_without_publishing(tmp_path: Path) -> None:
    source = tmp_path / "source.ckp"
    source.write_bytes(b"corrupt")
    destination = tmp_path / "model.ckp"

    result = _run_fetch(source, destination, "0" * 64)

    assert result.returncode != 0
    assert not destination.exists()


def test_fetch_script_refuses_to_replace_corrupt_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.ckp"
    source.write_bytes(b"correct")
    destination = tmp_path / "model.ckp"
    destination.write_bytes(b"existing but wrong")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    result = _run_fetch(source, destination, digest)

    assert result.returncode != 0
    assert destination.read_bytes() == b"existing but wrong"

