from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewTransform:
    scale: float
    offset_x: float
    offset_y: float

    def validate(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("预览缩放比必须为有限正数")
        if not math.isfinite(self.offset_x) or not math.isfinite(self.offset_y):
            raise ValueError("预览偏移无效")

    def image_to_view(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def view_to_image(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale


def clamp_transform(
    transform: ViewTransform,
    *,
    image_size: tuple[int, int],
    view_size: tuple[int, int],
) -> ViewTransform:
    transform.validate()
    image_width, image_height = image_size
    view_width, view_height = view_size
    if min(image_width, image_height, view_width, view_height) <= 0:
        raise ValueError("图像和窗口尺寸必须为正数")

    def clamp_axis(offset: float, image: int, view: int) -> float:
        extent = image * transform.scale
        if extent <= view:
            return (view - extent) / 2.0
        return min(0.0, max(view - extent, offset))

    return ViewTransform(
        transform.scale,
        clamp_axis(transform.offset_x, image_width, view_width),
        clamp_axis(transform.offset_y, image_height, view_height),
    )


def fit_transform(image_size: tuple[int, int], view_size: tuple[int, int]) -> ViewTransform:
    image_width, image_height = image_size
    view_width, view_height = view_size
    if min(image_width, image_height, view_width, view_height) <= 0:
        raise ValueError("图像和窗口尺寸必须为正数")
    scale = min(view_width / image_width, view_height / image_height)
    return clamp_transform(
        ViewTransform(scale, 0.0, 0.0), image_size=image_size, view_size=view_size
    )


def zoom_about(
    transform: ViewTransform,
    *,
    factor: float,
    anchor: tuple[float, float],
    image_size: tuple[int, int],
    view_size: tuple[int, int],
    minimum_scale: float,
    maximum_scale: float = 8.0,
) -> ViewTransform:
    if not math.isfinite(factor) or factor <= 0 or minimum_scale <= 0 or maximum_scale < minimum_scale:
        raise ValueError("预览缩放参数无效")
    image_x, image_y = transform.view_to_image(*anchor)
    scale = min(maximum_scale, max(minimum_scale, transform.scale * factor))
    candidate = ViewTransform(
        scale,
        anchor[0] - image_x * scale,
        anchor[1] - image_y * scale,
    )
    return clamp_transform(candidate, image_size=image_size, view_size=view_size)


def pan_transform(
    transform: ViewTransform,
    *,
    delta: tuple[float, float],
    image_size: tuple[int, int],
    view_size: tuple[int, int],
) -> ViewTransform:
    return clamp_transform(
        ViewTransform(transform.scale, transform.offset_x + delta[0], transform.offset_y + delta[1]),
        image_size=image_size,
        view_size=view_size,
    )


def clamp_split(position: float, view_width: int) -> float:
    if view_width <= 0 or not math.isfinite(position):
        raise ValueError("分割线参数无效")
    return min(float(view_width), max(0.0, position))
