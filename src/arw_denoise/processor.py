from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .denoise import HaarWaveletDenoiser
from .dnglab import DngLabClient, DngLabResult
from .pipeline import pack_normalized_bayer, tiled_inference, unpack_normalized_bayer
from .raw import RawDecoder, RawPyDecoder


@dataclass(frozen=True)
class ProcessingSettings:
    strength: float = 1.0
    tile_size: int = 1024
    overlap: int = 64


class CpuRawProcessor:
    def __init__(self, decoder: RawDecoder | None = None, dnglab: DngLabClient | None = None):
        self.decoder = decoder or RawPyDecoder()
        self.dnglab = dnglab or DngLabClient()
        self.denoiser = HaarWaveletDenoiser()

    def process(
        self,
        source: Path,
        output: Path,
        settings: ProcessingSettings | None = None,
        on_phase: Callable[[str], None] | None = None,
    ) -> DngLabResult:
        settings = settings or ProcessingSettings()
        frame = self.decoder.decode(source)
        if on_phase:
            on_phase("denoising")
        packed, context = pack_normalized_bayer(frame.pixels, frame.metadata)
        processed = tiled_inference(
            packed,
            lambda tile: self.denoiser.denoise(tile, strength=settings.strength),
            tile_size=settings.tile_size,
            overlap=settings.overlap,
        )
        restored = unpack_normalized_bayer(processed, context)
        if on_phase:
            on_phase("writing")
        return self.dnglab.write_processed_cfa(source, output, restored, frame.metadata)
