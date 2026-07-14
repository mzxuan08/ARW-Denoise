from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.release_manifest import collect_files, verify_manifest, write_manifest


def test_manifest_paths_are_normalized_sorted_and_hashed(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    (tmp_path / "z.txt").write_bytes(b"z")
    (tmp_path / "folder" / "a.txt").write_bytes(b"a")
    manifest_path, sums_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["files"]] == ["folder/a.txt", "z.txt"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert "folder/a.txt" in sums_path.read_text(encoding="utf-8")
    assert verify_manifest(tmp_path) == []


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra"])
def test_verify_rejects_every_bundle_mutation(tmp_path: Path, mutation: str) -> None:
    target = tmp_path / "app.exe"
    target.write_bytes(b"original")
    write_manifest(tmp_path)
    if mutation == "changed":
        target.write_bytes(b"modified")
    elif mutation == "missing":
        target.unlink()
    else:
        (tmp_path / "surprise.dll").write_bytes(b"extra")
    assert any(mutation in error or (mutation == "changed" and "changed" in error) for error in verify_manifest(tmp_path))


def test_collect_rejects_case_insensitive_duplicate_names(tmp_path: Path, monkeypatch) -> None:
    first, second = tmp_path / "A.dll", tmp_path / "a.dll"
    first.write_bytes(b"a")
    try:
        second.write_bytes(b"b")
    except OSError:
        pytest.skip("filesystem is case insensitive")
    if first.samefile(second):
        pytest.skip("filesystem is case insensitive")
    with pytest.raises(ValueError, match="duplicate"):
        collect_files(tmp_path)


def test_verify_rejects_changed_or_missing_checksum_index(tmp_path: Path) -> None:
    (tmp_path / "app.exe").write_bytes(b"app")
    _manifest, sums = write_manifest(tmp_path)
    sums.write_text("tampered\n", encoding="utf-8")
    assert verify_manifest(tmp_path) == ["changed: SHA256SUMS.txt"]
    sums.unlink()
    assert verify_manifest(tmp_path) == ["missing SHA256SUMS.txt"]
