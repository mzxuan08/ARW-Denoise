import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import tifffile

from arw_denoise.dnglab import DngLabClient
from arw_denoise.domain import ExternalToolError
from arw_denoise.domain import RawMetadata
from arw_denoise.task_control import CancellationToken, ProcessingCancelled


class FakeDngLab(DngLabClient):
    def __init__(self):
        self.executable = Path("fake-dnglab.exe")
        self.timeout_seconds = 5

    def _run(self, *args: str, cancellation=None) -> subprocess.CompletedProcess[str]:
        if args[0] == "convert":
            Path(args[-1]).write_bytes(b"fake dng")
            return subprocess.CompletedProcess(args, 0, "converted", "")
        if args[0] == "--version":
            return subprocess.CompletedProcess(args, 0, "dnglab 0.test", "")
        raise AssertionError(args)

    def analyze(self, path: Path, cancellation=None) -> dict:
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
    def _run(self, *args: str, cancellation=None) -> subprocess.CompletedProcess[str]:
        if args[0] == "convert":
            with tifffile.TiffWriter(args[-1]) as tif:
                tif.write(
                    np.zeros((4, 6, 3), np.uint8), photometric="rgb", metadata=None, subifds=1,
                    extratags=[
                        (50721, "2i", 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), False),
                        (50728, "2I", 3, (1, 1, 1, 1, 1, 1), False),
                        (50778, "H", 1, 21, False),
                    ],
                )
                tif.write(
                    np.full((8, 12), 512, np.uint16), photometric=32803,
                    compression=None, metadata=None,
                    extratags=[
                    (33421, "H", 2, (2, 2), False),
                    (33422, "B", 4, (0, 1, 1, 2), False),
                    (50713, "H", 2, (2, 2), False),
                    (50714, "H", 4, (512, 512, 512, 512), False),
                    (50717, "H", 1, 16383, False),
                    (50719, "H", 2, (0, 0), False),
                    (50720, "I", 2, (12, 8), False),
                    (50781, "B", 16, bytes(range(16)), False),
                    (50972, "B", 16, bytes(range(16)), False),
                    (51111, "B", 16, bytes(range(16)), False),
                    ],
                )
            return subprocess.CompletedProcess(args, 0, "converted", "")
        return super()._run(*args, cancellation=cancellation)

    def analyze(self, path: Path, cancellation=None) -> dict:
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
    with tifffile.TiffFile(output) as tif:
        np.testing.assert_array_equal(tif.series[1].asarray(), pixels)
    with tifffile.TiffFile(output) as tif:
        all_tags = {int(tag.code) for series in tif.series for page in series.pages for tag in page.tags}
        assert not ({50781, 50972, 51111} & all_tags)


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


def test_running_dnglab_process_is_terminated_when_cancelled(tmp_path: Path):
    token = CancellationToken()

    class Process:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False

        def communicate(self, timeout=None):
            if not self.terminated and not self.killed:
                token.cancel()
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = -15
            return ("", "")

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    process = Process()
    client = DngLabClient.__new__(DngLabClient)
    client.executable = tmp_path / "dnglab.exe"
    client.timeout_seconds = 5
    client.poll_seconds = 0.01
    client.popen_factory = lambda *args, **kwargs: process

    with pytest.raises(ProcessingCancelled):
        client._run("--version", cancellation=token)
    assert process.terminated is True
    assert process.killed is False


def test_cancelled_conversion_removes_only_its_temporary_file(tmp_path: Path):
    source = tmp_path / "sample.ARW"
    source.write_bytes(b"raw")
    output = tmp_path / "sample.dng"
    unrelated = tmp_path / ".unrelated.processing.dng"
    unrelated.write_bytes(b"keep")
    token = CancellationToken()

    class CancellingDngLab(FakeDngLab):
        def _run(self, *args: str, cancellation=None):
            if args[0] == "convert":
                Path(args[-1]).write_bytes(b"partial")
                token.cancel()
                token.check()
            return super()._run(*args, cancellation=cancellation)

    with pytest.raises(ProcessingCancelled):
        CancellingDngLab().compatibility_convert(source, output, cancellation=token)
    assert not output.exists()
    assert unrelated.read_bytes() == b"keep"
    assert list(tmp_path.glob(".sample.*.tmp.dng")) == []
