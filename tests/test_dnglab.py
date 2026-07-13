import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import tifffile

from arw_denoise.dnglab import DngLabClient
from arw_denoise.domain import ExternalToolError
from arw_denoise.domain import RawMetadata


class FakeDngLab(DngLabClient):
    def __init__(self):
        self.executable = Path("fake-dnglab.exe")
        self.timeout_seconds = 5

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        if args[0] == "convert":
            Path(args[-1]).write_bytes(b"fake dng")
            return subprocess.CompletedProcess(args, 0, "converted", "")
        if args[0] == "--version":
            return subprocess.CompletedProcess(args, 0, "dnglab 0.test", "")
        raise AssertionError(args)

    def analyze(self, path: Path) -> dict:
        assert path.read_bytes() == b"fake dng"
        return {"file": {"valid": True}}


def test_compatibility_convert_is_atomic_and_refuses_overwrite(tmp_path: Path):
    source = tmp_path / "sample.ARW"
    source.write_bytes(b"raw")
    output = tmp_path / "sample.dng"
    result = FakeDngLab().compatibility_convert(source, output)
    assert result.output == output.resolve()
    assert output.read_bytes() == b"fake dng"
    assert not list(tmp_path.glob("*.tmp.dng"))
    with pytest.raises(ExternalToolError, match="覆盖"):
        FakeDngLab().compatibility_convert(source, output)


class FakeCfaDngLab(FakeDngLab):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        if args[0] == "convert":
            tifffile.imwrite(
                args[-1], np.full((8, 12), 512, np.uint16), photometric=32803,
                compression=None, metadata=None,
                extratags=[
                    (33421, "H", 2, (2, 2), False),
                    (33422, "B", 4, (0, 1, 1, 2), False),
                    (50714, "H", 4, (512, 512, 512, 512), False),
                    (50717, "H", 1, 16383, False),
                ],
            )
            return subprocess.CompletedProcess(args, 0, "converted", "")
        return super()._run(*args)

    def analyze(self, path: Path) -> dict:
        assert path.stat().st_size > 0
        return {"file": {"valid": True}}


def test_write_processed_cfa_publishes_validated_pixels(tmp_path: Path):
    source = tmp_path / "sample.ARW"
    source.write_bytes(b"raw")
    output = tmp_path / "sample_DN.dng"
    pixels = np.arange(96, dtype=np.uint16).reshape(8, 12) + 512
    metadata = RawMetadata(
        path=source, width=12, height=8, raw_width=12, raw_height=8,
        cfa_pattern=(0, 1, 1, 2), color_description="RGBG",
        black_levels=(512, 512, 512, 512), white_level=16383, bits_per_sample=14,
    )
    FakeCfaDngLab().write_processed_cfa(source, output, pixels, metadata)
    np.testing.assert_array_equal(tifffile.imread(output), pixels)


def test_packaged_runtime_discovers_bundled_dnglab(tmp_path: Path, monkeypatch):
    executable = tmp_path / "tools" / "dnglab.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert DngLabClient._discover(None) == str(executable)


def test_atomic_publish_does_not_overwrite_racing_file(tmp_path: Path):
    temporary = tmp_path / ".temporary.dng"
    output = tmp_path / "output.dng"
    temporary.write_bytes(b"new")
    output.write_bytes(b"other process")
    with pytest.raises(ExternalToolError, match="已存在"):
        DngLabClient._publish_no_overwrite(temporary, output)
    assert output.read_bytes() == b"other process"
    assert temporary.read_bytes() == b"new"
