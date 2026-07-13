from __future__ import annotations

from pathlib import Path

import numpy as np

from .domain import ExternalToolError, RawMetadata


def replace_cfa_pixels_in_place(path: Path, processed_visible: np.ndarray, metadata: RawMetadata) -> None:
    """Replace only the CFA sample plane in an uncompressed DNG/TIFF.

    The base file must come from dnglab with uncompressed storage. Metadata and
    private IFDs remain byte-for-byte untouched. This function deliberately
    refuses compressed, tiled, ambiguous, or non-memory-mappable pixel planes.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ExternalToolError("缺少 tifffile，无法安全写回 CFA DNG") from exc

    path = Path(path)
    if processed_visible.shape != (metadata.height, metadata.width):
        raise ExternalToolError("处理后的 CFA 尺寸与 RAW active area 不一致")
    if not np.issubdtype(processed_visible.dtype, np.integer):
        raise ExternalToolError("处理后的 CFA 必须是整数采样")

    candidates: list[tuple[int, tuple[int, ...]]] = []
    try:
        with tifffile.TiffFile(path) as tif:
            for index, page in enumerate(tif.pages):
                photometric = page.tags.get(262)
                is_cfa = photometric is not None and int(photometric.value) == 32803
                shape = tuple(int(v) for v in page.shape)
                if is_cfa and page.dtype == processed_visible.dtype and shape in {
                    (metadata.raw_height, metadata.raw_width),
                    (metadata.height, metadata.width),
                }:
                    if page.compression.value != 1:
                        raise ExternalToolError("基础 DNG 的 CFA 平面不是无压缩格式")
                    candidates.append((index, shape))
    except ExternalToolError:
        raise
    except Exception as exc:
        raise ExternalToolError(f"无法读取基础 DNG 结构：{exc}") from exc

    if len(candidates) != 1:
        raise ExternalToolError(f"无法唯一定位 CFA 像素平面（候选数：{len(candidates)}）")
    page_index, shape = candidates[0]
    try:
        mapped = tifffile.memmap(path, page=page_index, mode="r+")
    except Exception as exc:
        raise ExternalToolError("CFA 平面不是连续可写存储；拒绝冒险修改 DNG") from exc
    try:
        if tuple(mapped.shape) != shape:
            raise ExternalToolError("映射后的 CFA 平面尺寸发生变化")
        if shape == (metadata.height, metadata.width):
            mapped[:, :] = processed_visible
        else:
            top = metadata.top_margin
            left = metadata.left_margin
            mapped[top:top + metadata.height, left:left + metadata.width] = processed_visible
        mapped.flush()
    finally:
        del mapped

