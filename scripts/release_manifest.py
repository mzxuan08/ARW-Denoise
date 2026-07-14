from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path


MANIFEST_NAME = "release-manifest.json"
SUMS_NAME = "SHA256SUMS.txt"
GENERATED_NAMES = frozenset({MANIFEST_NAME, SUMS_NAME})


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[dict[str, object]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_symlink():
            raise ValueError(f"Release bundle cannot contain symlinks: {path}")
        if not path.is_file() or path.name in GENERATED_NAMES:
            continue
        relative = path.relative_to(root).as_posix()
        folded = relative.casefold()
        if folded in seen:
            raise ValueError(f"Case-insensitive duplicate paths: {seen[folded]} / {relative}")
        seen[folded] = relative
        records.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def build_manifest(root: Path) -> dict[str, object]:
    return {"schema_version": 1, "files": collect_files(root)}


def write_manifest(root: Path) -> tuple[Path, Path]:
    root = Path(root).resolve()
    manifest = build_manifest(root)
    manifest_path = root / MANIFEST_NAME
    sums_path = root / SUMS_NAME
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    sums_text = "".join(
        f"{record['sha256']}  {record['path']}\n" for record in manifest["files"]  # type: ignore[index]
    )
    for path, content in ((manifest_path, manifest_text), (sums_path, sums_text)):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8", newline="\n")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest_path, sums_path


def verify_manifest(root: Path) -> list[str]:
    root = Path(root).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing {MANIFEST_NAME}"]
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected.get("schema_version") != 1 or not isinstance(expected.get("files"), list):
            return ["invalid manifest schema"]
        actual = build_manifest(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]
    expected_by_path = {record.get("path"): record for record in expected["files"]}
    actual_by_path = {record.get("path"): record for record in actual["files"]}
    errors: list[str] = []
    for path in sorted(expected_by_path.keys() - actual_by_path.keys()):
        errors.append(f"missing: {path}")
    for path in sorted(actual_by_path.keys() - expected_by_path.keys()):
        errors.append(f"extra: {path}")
    for path in sorted(expected_by_path.keys() & actual_by_path.keys()):
        if expected_by_path[path] != actual_by_path[path]:
            errors.append(f"changed: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a deterministic release manifest")
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        errors = verify_manifest(args.distribution)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"Release manifest verified: {args.distribution.resolve()}")
        return 0
    manifest, sums = write_manifest(args.distribution)
    print(manifest)
    print(sums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
