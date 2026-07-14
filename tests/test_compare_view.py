from __future__ import annotations

import pytest

from arw_denoise.compare_view import (
    ViewTransform,
    clamp_split,
    fit_transform,
    pan_transform,
    zoom_about,
)


def test_fit_centers_image_and_mapping_is_reversible() -> None:
    transform = fit_transform((4000, 3000), (1000, 1000))
    assert transform.scale == pytest.approx(0.25)
    assert transform.offset_y == pytest.approx(125)
    point = (1234.5, 987.5)
    assert transform.view_to_image(*transform.image_to_view(*point)) == pytest.approx(point)


def test_zoom_keeps_anchor_on_same_image_pixel_when_not_clamped() -> None:
    original = fit_transform((4000, 3000), (1000, 700))
    anchor = (480.0, 360.0)
    before = original.view_to_image(*anchor)
    zoomed = zoom_about(
        original,
        factor=2,
        anchor=anchor,
        image_size=(4000, 3000),
        view_size=(1000, 700),
        minimum_scale=original.scale,
    )
    assert zoomed.view_to_image(*anchor) == pytest.approx(before)


def test_pan_and_split_are_clamped_to_visible_bounds() -> None:
    transform = ViewTransform(1.0, 0.0, 0.0)
    panned = pan_transform(
        transform, delta=(10000, -10000), image_size=(2000, 1200), view_size=(800, 600)
    )
    assert panned.offset_x == 0
    assert panned.offset_y == -600
    assert clamp_split(-5, 800) == 0
    assert clamp_split(900, 800) == 800
