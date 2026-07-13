from pathlib import Path

from arw_denoise.naming import available_output_path


def test_available_output_path_never_overwrites(tmp_path: Path):
    source = tmp_path / "DSC00001.ARW"
    output = tmp_path / "out"
    output.mkdir()
    assert available_output_path(source, output).name == "DSC00001_DN.dng"
    (output / "DSC00001_DN.dng").touch()
    assert available_output_path(source, output).name == "DSC00001_DN_2.dng"

