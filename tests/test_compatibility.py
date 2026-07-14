from __future__ import annotations

import json
from pathlib import Path

from arw_denoise.compatibility import (
    CompatibilityResult,
    discover_arw_files,
    scan_arw_files,
    source_snapshot,
    write_compatibility_report,
)
from arw_denoise.domain import RawMetadata, UnsupportedRawError


def metadata(path: Path, *, model: str = "ILCE-7CM2") -> RawMetadata:
    return RawMetadata(
        path=path,
        width=7032,
        height=4688,
        raw_width=7168,
        raw_height=5120,
        cfa_pattern=(0, 1, 3, 2),
        color_description="RGBG",
        black_levels=(512, 512, 512, 512),
        white_level=15360,
        bits_per_sample=16,
        make="Sony",
        model=model,
        iso=400,
    )


class FakeDecoder:
    def probe(self, path: Path) -> RawMetadata:
        if path.name.startswith("bad"):
            raise UnsupportedRawError("CFA 不受支持")
        return metadata(path, model="ILCE-1" if path.name.startswith("one") else "ILCE-7CM2")


def test_recursive_discovery_finds_only_arw_in_stable_order(tmp_path: Path) -> None:
    nested = tmp_path / "2026" / "旅行"
    nested.mkdir(parents=True)
    first = tmp_path / "A.ARW"
    second = nested / "b.arw"
    first.write_bytes(b"raw")
    second.write_bytes(b"raw")
    (nested / "ignore.jpg").write_bytes(b"jpg")

    assert discover_arw_files(tmp_path) == [first.resolve(), second.resolve()]
    assert discover_arw_files(tmp_path, recursive=False) == [first.resolve()]


def test_parallel_scan_preserves_input_order_and_isolates_failures(tmp_path: Path) -> None:
    paths = [tmp_path / "one.ARW", tmp_path / "bad.ARW", tmp_path / "three.ARW"]
    results = scan_arw_files(paths, decoder_factory=FakeDecoder, max_workers=2)

    assert [result.path for result in results] == [path.resolve() for path in paths]
    assert [result.supported for result in results] == [True, False, True]
    assert results[0].metadata is not None
    assert results[0].metadata.model == "ILCE-1"
    assert results[1].error == "CFA 不受支持"


def test_source_snapshot_contains_queue_display_fields(tmp_path: Path) -> None:
    snapshot = source_snapshot(metadata(tmp_path / "a.ARW"))
    assert snapshot == {
        "make": "Sony",
        "model": "ILCE-7CM2",
        "iso": 400,
        "width": 7032,
        "height": 4688,
        "bits_per_sample": 16,
        "cfa": "RGGB",
    }


def test_report_is_utf8_atomic_and_summarizes_results(tmp_path: Path) -> None:
    results = [
        CompatibilityResult(tmp_path / "好.ARW", metadata=metadata(tmp_path / "好.ARW")),
        CompatibilityResult(tmp_path / "坏.ARW", error="无法识别 CFA"),
    ]
    target = tmp_path / "reports" / "compatibility.json"

    write_compatibility_report(results, target)

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["summary"] == {"total": 2, "supported": 1, "unsupported": 1}
    assert document["files"][0]["camera"]["model"] == "ILCE-7CM2"
    assert document["files"][1]["error"] == "无法识别 CFA"
    assert not target.with_suffix(".json.tmp").exists()
