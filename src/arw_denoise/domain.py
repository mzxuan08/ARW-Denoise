from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ArwDenoiseError(RuntimeError):
    """Base error with a message suitable for the UI."""


class UnsupportedRawError(ArwDenoiseError):
    """The source RAW cannot be processed without guessing metadata."""


class ExternalToolError(ArwDenoiseError):
    """An external helper failed or produced an invalid result."""


@dataclass(frozen=True)
class RawMetadata:
    path: Path
    width: int
    height: int
    raw_width: int
    raw_height: int
    cfa_pattern: tuple[int, int, int, int]
    color_description: str
    black_levels: tuple[float, float, float, float]
    white_level: float
    bits_per_sample: int
    top_margin: int = 0
    left_margin: int = 0
    make: str | None = None
    model: str | None = None
    iso: int | None = None
    shutter_seconds: float | None = None
    aperture: float | None = None
    focal_length_mm: float | None = None

    @property
    def resolved_cfa(self) -> tuple[str, str, str, str]:
        description = self.color_description.upper()
        try:
            return tuple(description[index] for index in self.cfa_pattern)  # type: ignore[return-value]
        except (IndexError, TypeError) as exc:
            raise UnsupportedRawError("CFA 索引无法映射到颜色描述") from exc

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise UnsupportedRawError("RAW 的可见区域尺寸无效")
        if self.width % 2 or self.height % 2:
            raise UnsupportedRawError("首版只支持偶数尺寸的 2x2 Bayer RAW")
        if self.top_margin < 0 or self.left_margin < 0:
            raise UnsupportedRawError("RAW active area 偏移无效")
        if self.top_margin + self.height > self.raw_height or self.left_margin + self.width > self.raw_width:
            raise UnsupportedRawError("RAW active area 超出完整像素平面")
        if len(self.cfa_pattern) != 4 or any(v not in (0, 1, 2, 3) for v in self.cfa_pattern):
            raise UnsupportedRawError("无法可靠识别 2x2 Bayer CFA")
        colors = self.resolved_cfa
        if colors.count("R") != 1 or colors.count("G") != 2 or colors.count("B") != 1:
            raise UnsupportedRawError(f"不支持的 CFA 颜色排列：{''.join(colors)}")
        if len(self.black_levels) != 4:
            raise UnsupportedRawError("缺少四通道黑电平")
        if self.white_level <= max(self.black_levels):
            raise UnsupportedRawError("RAW 白电平不高于黑电平")
        if not 8 <= self.bits_per_sample <= 16:
            raise UnsupportedRawError("RAW 位深不在支持范围 8–16 bit")


@dataclass
class RawFrame:
    metadata: RawMetadata
    pixels: "object"

    def validate(self) -> None:
        self.metadata.validate()
        shape: Sequence[int] | None = getattr(self.pixels, "shape", None)
        if shape is None or tuple(shape) != (self.metadata.height, self.metadata.width):
            raise UnsupportedRawError("RAW 像素尺寸与元数据不一致")
