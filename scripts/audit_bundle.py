from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PRUNE_PATHS = (
    "_internal/_tcl_data",
    "_internal/_tk_data",
    "_internal/tcl8",
    "_internal/onnxruntime/capi/onnxruntime_providers_tensorrt.dll",
    "_internal/PySide6/plugins/generic",
    "_internal/PySide6/plugins/iconengines",
    "_internal/PySide6/plugins/networkinformation",
    "_internal/PySide6/plugins/platforminputcontexts",
    "_internal/PySide6/plugins/tls",
    "_internal/PySide6/plugins/imageformats",
    "_internal/PySide6/plugins/platforms/qdirect2d.dll",
    "_internal/PySide6/plugins/platforms/qminimal.dll",
    "_internal/PySide6/plugins/platforms/qoffscreen.dll",
    "_internal/PySide6/opengl32sw.dll",
    "_internal/PySide6/Qt6Network.dll",
    "_internal/PySide6/Qt6OpenGL.dll",
    "_internal/PySide6/Qt6Pdf.dll",
    "_internal/PySide6/Qt6Qml.dll",
    "_internal/PySide6/Qt6QmlMeta.dll",
    "_internal/PySide6/Qt6QmlModels.dll",
    "_internal/PySide6/Qt6QmlWorkerScript.dll",
    "_internal/PySide6/Qt6Quick.dll",
    "_internal/PySide6/Qt6Svg.dll",
    "_internal/PySide6/Qt6VirtualKeyboard.dll",
)


def _contained(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / Path(relative)).resolve()
    candidate.relative_to(root)
    return candidate


def bundle_report(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    files = [path for path in root.rglob("*") if path.is_file()]
    groups: dict[str, int] = {}
    largest: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) == 1 else "/".join(relative.parts[:2])
        groups[group] = groups.get(group, 0) + path.stat().st_size
        largest.append({"path": relative.as_posix(), "size": path.stat().st_size})
    largest.sort(key=lambda item: int(item["size"]), reverse=True)
    candidates = [relative for relative in PRUNE_PATHS if _contained(root, relative).exists()]
    return {
        "schema_version": 1,
        "total_bytes": sum(path.stat().st_size for path in files),
        "file_count": len(files),
        "groups": dict(sorted(groups.items())),
        "largest_files": largest[:30],
        "prune_candidates": candidates,
    }


def prune_bundle(root: Path) -> list[str]:
    root = Path(root).resolve()
    if not root.is_dir() or not (root / "ArwDenoise.exe").is_file():
        raise ValueError("Refusing to prune a directory that is not an ArwDenoise distribution")
    removed: list[str] = []
    for relative in PRUNE_PATHS:
        target = _contained(root, relative)
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(relative)
        elif target.is_file():
            target.unlink()
            removed.append(relative)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and safely prune known-unused bundle assets")
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    before = bundle_report(args.distribution)
    removed = prune_bundle(args.distribution) if args.prune else []
    payload = {"before": before, "removed": removed, "after": bundle_report(args.distribution)}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
