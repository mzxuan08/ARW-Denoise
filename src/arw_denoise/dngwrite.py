from __future__ import annotations

from pathlib import Path

import numpy as np

from .domain import ExternalToolError, RawMetadata


_PIXEL_LOCATION_TAGS = {273, 279, 324, 325}


def snapshot_dng_metadata(path: Path) -> tuple[tuple[int, int, str, int, str], ...]:
    """Canonical snapshot of every TIFF/DNG metadata tag, excluding pixel offsets."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            return tuple(
                (page_index, int(tag.code), str(tag.dtype), int(tag.count), repr(tag.value))
                for page_index, page in enumerate(tif.pages)
                for tag in page.tags.values()
                if int(tag.code) not in _PIXEL_LOCATION_TAGS
            )
    except Exception as exc:
        raise ExternalToolError(f"无法建立 DNG 元数据快照：{exc}") from exc


def _numeric_tag(page, code: int, label: str, count: int | None = None) -> np.ndarray:
    tag = page.tags.get(code)
    if tag is None:
        raise ExternalToolError(f"DNG 缺少 {label}")
    try:
        values = np.asarray(tag.value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ExternalToolError(f"DNG {label} 不是有效数值") from exc
    if count is not None and values.size != count:
        raise ExternalToolError(f"DNG {label} 数量应为 {count}")
    if not np.all(np.isfinite(values)):
        raise ExternalToolError(f"DNG {label} 包含无效数值")
    return values


def _validate_editability_tags(page, metadata: RawMetadata, shape: tuple[int, ...]) -> None:
    repeat = tuple(int(v) for v in _numeric_tag(page, 50713, "BlackLevelRepeatDim", 2))
    black = _numeric_tag(page, 50714, "BlackLevel")
    if black.size == 4 and repeat != (2, 2):
        raise ExternalToolError("DNG BlackLevelRepeatDim 无法表示四位置黑电平")
    origin = _numeric_tag(page, 50719, "DefaultCropOrigin", 2)
    size = _numeric_tag(page, 50720, "DefaultCropSize", 2)
    active_width, active_height = metadata.width, metadata.height
    if np.any(origin < 0) or np.any(size <= 0) or origin[0] + size[0] > active_width or origin[1] + size[1] > active_height:
        raise ExternalToolError("DNG 默认裁剪超出 ActiveArea")
    matrix = _numeric_tag(page, 50721, "ColorMatrix1", 9)
    if np.linalg.matrix_rank(matrix.reshape(3, 3)) < 3:
        raise ExternalToolError("DNG ColorMatrix1 不可逆")
    illuminant = _numeric_tag(page, 50778, "CalibrationIlluminant1", 1)
    if illuminant[0] <= 0:
        raise ExternalToolError("DNG CalibrationIlluminant1 无效")
    neutral = _numeric_tag(page, 50728, "AsShotNeutral", 3)
    if np.any(neutral <= 0):
        raise ExternalToolError("DNG AsShotNeutral 无效")
    orientation = page.tags.get(274)
    orientation_value = int(orientation.value) if orientation is not None else 1
    if orientation_value not in range(1, 9):
        raise ExternalToolError("DNG Orientation 无效")


def _find_cfa_page(tif, metadata: RawMetadata, dtype: np.dtype) -> tuple[int, tuple[int, ...]]:
    candidates: list[tuple[int, tuple[int, ...]]] = []
    for index, page in enumerate(tif.pages):
        photometric = page.tags.get(262)
        is_cfa = photometric is not None and int(photometric.value) == 32803
        shape = tuple(int(v) for v in page.shape)
        if is_cfa and page.dtype == dtype and shape in {
            (metadata.raw_height, metadata.raw_width),
            (metadata.height, metadata.width),
        }:
            if page.compression.value != 1:
                raise ExternalToolError("基础 DNG 的 CFA 平面不是无压缩格式")
            candidates.append((index, shape))
    if len(candidates) != 1:
        raise ExternalToolError(f"无法唯一定位 CFA 像素平面（候选数：{len(candidates)}）")
    return candidates[0]


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

    try:
        with tifffile.TiffFile(path) as tif:
            page_index, shape = _find_cfa_page(tif, metadata, processed_visible.dtype)
    except ExternalToolError:
        raise
    except Exception as exc:
        raise ExternalToolError(f"无法读取基础 DNG 结构：{exc}") from exc

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


def validate_processed_dng(path: Path, processed_visible: np.ndarray, metadata: RawMetadata) -> None:
    """Validate semantic DNG tags and exact written CFA samples before publish."""
    try:
        import tifffile
    except ImportError as exc:
        raise ExternalToolError("缺少 tifffile，无法校验 CFA DNG") from exc
    try:
        with tifffile.TiffFile(path) as tif:
            page_index, shape = _find_cfa_page(tif, metadata, processed_visible.dtype)
            page = tif.pages[page_index]
            _validate_editability_tags(page, metadata, shape)
            if int(page.bitspersample) < metadata.bits_per_sample or int(page.bitspersample) > 16:
                raise ExternalToolError("DNG BitsPerSample 无法容纳源 RAW 位深")
            repeat = page.tags.get(33421)
            pattern = page.tags.get(33422)
            if repeat is None or tuple(int(v) for v in repeat.value) != (2, 2):
                raise ExternalToolError("DNG CFARepeatPatternDim 与源 RAW 不一致")
            expected_codes = tuple({"R": 0, "G": 1, "B": 2}[color] for color in metadata.resolved_cfa)
            if pattern is None or tuple(int(v) for v in pattern.value) != expected_codes:
                raise ExternalToolError("DNG CFAPattern 与源 RAW 不一致")
            expected_active = (
                (0, 0, metadata.height, metadata.width)
                if shape == (metadata.height, metadata.width)
                else (
                    metadata.top_margin,
                    metadata.left_margin,
                    metadata.top_margin + metadata.height,
                    metadata.left_margin + metadata.width,
                )
            )
            active = page.tags.get(50829)
            actual_active = tuple(int(v) for v in active.value) if active is not None else (0, 0, shape[0], shape[1])
            if actual_active != expected_active:
                raise ExternalToolError("DNG ActiveArea 与源 RAW 不一致")
            for tag_code, expected, label in ((271, metadata.make, "Make"), (272, metadata.model, "Model")):
                tag = page.tags.get(tag_code)
                if expected and (tag is None or str(tag.value).strip() != expected.strip()):
                    raise ExternalToolError(f"DNG {label} 与源 RAW 不一致")
            black = page.tags.get(50714)
            white = page.tags.get(50717)
            if black is None or white is None:
                raise ExternalToolError("DNG 缺少 BlackLevel 或 WhiteLevel")
            black_values = np.asarray(black.value, dtype=np.float64).reshape(-1)
            expected_black = np.asarray(metadata.black_levels)[np.asarray(metadata.cfa_pattern)]
            if black_values.size not in (1, 4):
                raise ExternalToolError("DNG BlackLevel 数量不受支持")
            if black_values.size == 4 and not np.allclose(black_values, expected_black, atol=1.0):
                raise ExternalToolError("DNG BlackLevel 与源 RAW 不一致")
            if black_values.size == 1 and not np.allclose(expected_black, black_values[0], atol=1.0):
                raise ExternalToolError("DNG 单一 BlackLevel 无法表示源 RAW 通道值")
            white_values = np.asarray(white.value, dtype=np.float64).reshape(-1)
            if not np.allclose(white_values, metadata.white_level, atol=1.0):
                raise ExternalToolError("DNG WhiteLevel 与源 RAW 不一致")
            full = page.asarray()
            if shape == (metadata.height, metadata.width):
                visible = full
            else:
                top, left = metadata.top_margin, metadata.left_margin
                visible = full[top:top + metadata.height, left:left + metadata.width]
            if not np.array_equal(visible, processed_visible):
                raise ExternalToolError("DNG CFA 像素回读结果与降噪输出不一致")
    except ExternalToolError:
        raise
    except Exception as exc:
        raise ExternalToolError(f"DNG 语义校验失败：{exc}") from exc
