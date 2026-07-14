from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from arw_denoise.preview import PreviewPair
from arw_denoise.preview_window import CompareCanvas


def test_compare_canvas_uses_one_synchronized_transform() -> None:
    app = QApplication.instance() or QApplication([])
    canvas = CompareCanvas()
    canvas.resize(800, 600)
    pair = PreviewPair(
        np.zeros((300, 400, 3), np.uint8), np.full((300, 400, 3), 128, np.uint8)
    )
    canvas.set_pair(pair)
    assert canvas.source_image is not None
    assert canvas.denoised_image is not None
    assert canvas.transform is not None
    assert canvas.transform.scale == pytest.approx(2.0)
    canvas.close()
    app.processEvents()
