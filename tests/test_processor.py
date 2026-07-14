from pathlib import Path

import numpy as np

from arw_denoise.dnglab import DngLabResult
from arw_denoise.domain import RawFrame, RawMetadata
from arw_denoise.processor import CpuRawProcessor, ProcessingSettings


class FakeDecoder:
    def __init__(self, pixels: np.ndarray, metadata: RawMetadata):
        self.frame = RawFrame(metadata, pixels)

    def probe(self, path: Path) -> RawMetadata:
        return self.frame.metadata

    def decode(self, path: Path) -> RawFrame:
        return self.frame


class FakeWriter:
    def __init__(self):
        self.pixels = None

    def write_processed_cfa(self, source, output, pixels, metadata, cancellation=None):
        self.pixels = pixels.copy()
        return DngLabResult(Path(output), "fake", {"valid": True})


def test_cpu_processor_keeps_integer_cfa_and_reduces_noise(tmp_path: Path):
    rng = np.random.default_rng(2)
    clean = np.full((64, 64), 4000, dtype=np.float32)
    pixels = np.clip(clean + rng.normal(0, 350, clean.shape), 512, 16383).astype(np.uint16)
    metadata = RawMetadata(
        path=tmp_path / "sample.ARW", width=64, height=64, raw_width=64, raw_height=64,
        cfa_pattern=(0, 1, 1, 2), color_description="RGBG",
        black_levels=(512, 512, 512, 512), white_level=16383, bits_per_sample=14,
    )
    writer = FakeWriter()
    processor = CpuRawProcessor(decoder=FakeDecoder(pixels, metadata), dnglab=writer)
    result = processor.process(metadata.path, tmp_path / "out.dng", ProcessingSettings(tile_size=64, overlap=8))
    assert result.version == "fake"
    assert writer.pixels.dtype == np.uint16
    assert writer.pixels.shape == pixels.shape
    assert float(np.var(writer.pixels.astype(float) - clean)) < float(np.var(pixels.astype(float) - clean))
