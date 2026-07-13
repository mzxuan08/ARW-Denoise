from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from .dnglab import DngLabClient
from .domain import RawFrame, RawMetadata, UnsupportedRawError


class RawDecoder(Protocol):
    def probe(self, path: Path) -> RawMetadata: ...
    def decode(self, path: Path) -> RawFrame: ...


def _bits_for_white_level(white_level: float) -> int:
    return max(8, min(16, int(math.ceil(math.log2(float(white_level) + 1.0)))))


class RawPyDecoder:
    """Thin, optional LibRaw/rawpy adapter that preserves the mosaic."""

    def __init__(self, allow_experimental: bool = False, dnglab: DngLabClient | None = None):
        self.allow_experimental = allow_experimental
        self.dnglab = dnglab

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
                dnglab = self.dnglab or DngLabClient()
                document = dnglab.metadata(path)
                try:
                    metadata_root = document["data"]["metadata"]
                    raw_params = metadata_root["rawParams"]
                    raw_metadata = metadata_root["rawMetadata"]
                    exif = raw_metadata["exif"]
                    active = raw_params["activeArea"]
                    active_point = active["p"]
                    active_size = active["d"]
                except (KeyError, TypeError) as exc:
                    raise UnsupportedRawError("dnglab 返回的 RAW 元数据结构不完整") from exc
                pattern_array = raw.raw_pattern
                if pattern_array is None or tuple(pattern_array.shape) != (2, 2):
                    raise UnsupportedRawError("首版仅支持 2x2 Bayer RAW")
                pattern = tuple(int(v) for v in pattern_array.reshape(-1))
                desc = bytes(raw.color_desc).decode("ascii", errors="replace").rstrip("\x00")
                sizes = raw.sizes
                black_levels = tuple(_rational(value) for value in raw_params["blacklevels"]["levels"])
                white_levels = raw_params.get("whitelevels") or [raw.white_level]
                result = RawMetadata(
                    path=path.resolve(),
                    width=int(active_size["w"]),
                    height=int(active_size["h"]),
                    raw_width=int(raw_params.get("rawWidth", sizes.raw_width)),
                    raw_height=int(raw_params.get("rawHeight", sizes.raw_height)),
                    cfa_pattern=pattern,  # type: ignore[arg-type]
                    color_description=desc,
                    black_levels=black_levels,  # type: ignore[arg-type]
                    white_level=float(white_levels[0]),
                    bits_per_sample=int(raw_params.get("bitDepth") or _bits_for_white_level(white_levels[0])),
                    top_margin=int(active_point["y"]),
                    left_margin=int(active_point["x"]),
                    make=raw_metadata.get("make"),
                    model=raw_metadata.get("model"),
                    iso=int(exif.get("iso_speed_ratings") or exif.get("recommended_exposure_index")) if (exif.get("iso_speed_ratings") or exif.get("recommended_exposure_index")) else None,
                    shutter_seconds=_optional_rational(exif.get("exposure_time")),
                    aperture=_optional_rational(exif.get("fnumber")),
                    focal_length_mm=_optional_rational(exif.get("focal_length")),
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
                top, left = metadata.top_margin, metadata.left_margin
                pixels = raw.raw_image[
                    top:top + metadata.height,
                    left:left + metadata.width,
                ].copy()
        except Exception as exc:
            raise UnsupportedRawError(f"无法解码 {Path(path).name} 的 Bayer 数据：{exc}") from exc
        frame = RawFrame(metadata=metadata, pixels=pixels)
        frame.validate()
        return frame


def _rational(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    numerator, separator, denominator = str(value).partition("/")
    if not separator:
        return float(numerator)
    return float(numerator) / float(denominator)


def _optional_rational(value: object | None) -> float | None:
    if value is None:
        return None
    return _rational(value)
