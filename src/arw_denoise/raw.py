from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from .domain import RawFrame, RawMetadata, UnsupportedRawError


class RawDecoder(Protocol):
    def probe(self, path: Path) -> RawMetadata: ...
    def decode(self, path: Path) -> RawFrame: ...


def _bits_for_white_level(white_level: float) -> int:
    return max(8, min(16, int(math.ceil(math.log2(float(white_level) + 1.0)))))


class RawPyDecoder:
    """Thin, optional LibRaw/rawpy adapter that preserves the mosaic."""

    def __init__(self, allow_experimental: bool = False):
        self.allow_experimental = allow_experimental

    @staticmethod
    def _rawpy():
        try:
            import rawpy  # type: ignore
        except ImportError as exc:
            raise UnsupportedRawError(
                "缺少 rawpy；请安装 arw-denoise[raw] 后再读取 ARW"
            ) from exc
        return rawpy

    def probe(self, path: Path) -> RawMetadata:
        rawpy = self._rawpy()
        path = Path(path)
        if not path.is_file():
            raise UnsupportedRawError(f"找不到 RAW 文件：{path.name}")
        try:
            with rawpy.imread(str(path)) as raw:
                pattern_array = raw.raw_pattern
                if pattern_array is None or tuple(pattern_array.shape) != (2, 2):
                    raise UnsupportedRawError("首版仅支持 2x2 Bayer RAW")
                pattern = tuple(int(v) for v in pattern_array.reshape(-1))
                desc = bytes(raw.color_desc).decode("ascii", errors="replace").rstrip("\x00")
                sizes = raw.sizes
                metadata = raw.metadata
                result = RawMetadata(
                    path=path.resolve(),
                    width=int(sizes.width),
                    height=int(sizes.height),
                    raw_width=int(sizes.raw_width),
                    raw_height=int(sizes.raw_height),
                    cfa_pattern=pattern,  # type: ignore[arg-type]
                    color_description=desc,
                    black_levels=tuple(float(v) for v in raw.black_level_per_channel),  # type: ignore[arg-type]
                    white_level=float(raw.white_level),
                    bits_per_sample=_bits_for_white_level(raw.white_level),
                    top_margin=int(sizes.top_margin),
                    left_margin=int(sizes.left_margin),
                    make=getattr(metadata, "make", None),
                    model=getattr(metadata, "model", None),
                    iso=int(metadata.iso_speed) if getattr(metadata, "iso_speed", 0) else None,
                    shutter_seconds=float(metadata.shutter) if getattr(metadata, "shutter", 0) else None,
                    aperture=float(metadata.aperture) if getattr(metadata, "aperture", 0) else None,
                    focal_length_mm=float(metadata.focal_len) if getattr(metadata, "focal_len", 0) else None,
                )
                result.validate()
                make = (result.make or "").strip().lower()
                model = (result.model or "").strip().upper()
                supported = "sony" in make and model == "ILCE-7CM2"
                if not supported and not self.allow_experimental:
                    label = " ".join(value for value in (result.make, result.model) if value) or "未知机型"
                    raise UnsupportedRawError(f"首版仅正式支持 Sony ILCE-7CM2；当前文件为 {label}")
                return result
        except UnsupportedRawError:
            raise
        except Exception as exc:
            raise UnsupportedRawError(f"LibRaw 无法读取 {path.name}：{exc}") from exc

    def decode(self, path: Path) -> RawFrame:
        rawpy = self._rawpy()
        metadata = self.probe(path)
        try:
            with rawpy.imread(str(path)) as raw:
                pixels = raw.raw_image_visible.copy()
        except Exception as exc:
            raise UnsupportedRawError(f"无法解码 {Path(path).name} 的 Bayer 数据：{exc}") from exc
        frame = RawFrame(metadata=metadata, pixels=pixels)
        frame.validate()
        return frame
