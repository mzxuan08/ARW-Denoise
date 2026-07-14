from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .compare_view import ViewTransform, clamp_transform, fit_transform, pan_transform, zoom_about
from .preview import PreviewPair
from .preview_cache import PreviewCache


def _qimage(value: np.ndarray) -> QImage:
    height, width, _channels = value.shape
    return QImage(
        value.data, width, height, value.strides[0], QImage.Format.Format_RGB888
    ).copy()


class PreviewLoader(QObject):
    ready = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(
        self,
        request_id: int,
        cache: PreviewCache,
        source: Path,
        denoised: Path,
        half_size: bool,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.cache = cache
        self.source = source
        self.denoised = denoised
        self.half_size = half_size
        self.cancelled = False

    @Slot()
    def run(self) -> None:
        try:
            if self.cancelled:
                return
            pair = self.cache.get_or_render(
                self.source, self.denoised, half_size=self.half_size
            )
            if not self.cancelled:
                self.ready.emit(self.request_id, pair)
        except Exception as exc:
            if not self.cancelled:
                self.failed.emit(self.request_id, f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def cancel(self) -> None:
        self.cancelled = True


class CompareCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 420)
        self.setMouseTracking(True)
        self.source_image: QImage | None = None
        self.denoised_image: QImage | None = None
        self.transform: ViewTransform | None = None
        self.split_ratio = 0.5
        self._drag_mode: str | None = None
        self._last_position = QPointF()
        self._fit_mode = True

    def set_pair(self, pair: PreviewPair, *, one_to_one: bool = False) -> None:
        pair.validate()
        self.source_image = _qimage(pair.source)
        self.denoised_image = _qimage(pair.denoised)
        if one_to_one:
            self.show_one_to_one()
        else:
            self.fit_image()

    def _sizes(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if self.source_image is None:
            return None
        return (self.source_image.width(), self.source_image.height()), (self.width(), self.height())

    def fit_image(self) -> None:
        sizes = self._sizes()
        if sizes is None or min(*sizes[1]) <= 0:
            return
        self.transform = fit_transform(*sizes)
        self._fit_mode = True
        self.update()

    def show_one_to_one(self) -> None:
        sizes = self._sizes()
        if sizes is None:
            return
        image_size, view_size = sizes
        self.transform = clamp_transform(
            ViewTransform(1.0, (view_size[0] - image_size[0]) / 2, (view_size[1] - image_size[1]) / 2),
            image_size=image_size,
            view_size=view_size,
        )
        self._fit_mode = False
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191d"))
        if self.source_image is None or self.denoised_image is None or self.transform is None:
            painter.setPen(QColor("#c8c8c8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "正在生成离线预览…")
            return
        target = QRectF(
            self.transform.offset_x,
            self.transform.offset_y,
            self.source_image.width() * self.transform.scale,
            self.source_image.height() * self.transform.scale,
        )
        split = self.width() * self.split_ratio
        painter.save()
        painter.setClipRect(QRectF(0, 0, split, self.height()))
        painter.drawImage(target, self.source_image)
        painter.restore()
        painter.save()
        painter.setClipRect(QRectF(split, 0, self.width() - split, self.height()))
        painter.drawImage(target, self.denoised_image)
        painter.restore()
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(int(split), 0, int(split), self.height())
        painter.fillRect(QRectF(12, 12, 70, 28), QColor(0, 0, 0, 150))
        painter.fillRect(QRectF(self.width() - 94, 12, 82, 28), QColor(0, 0, 0, 150))
        painter.setPen(QColor("white"))
        painter.drawText(QRectF(12, 12, 70, 28), Qt.AlignmentFlag.AlignCenter, "源 ARW")
        painter.drawText(
            QRectF(self.width() - 94, 12, 82, 28), Qt.AlignmentFlag.AlignCenter, "降噪 DNG"
        )

    def resizeEvent(self, _event) -> None:
        if self._fit_mode:
            self.fit_image()
        elif self.transform is not None and self._sizes() is not None:
            self.transform = clamp_transform(
                self.transform, image_size=self._sizes()[0], view_size=self._sizes()[1]
            )

    def wheelEvent(self, event: QWheelEvent) -> None:
        sizes = self._sizes()
        if sizes is None or self.transform is None:
            return
        fit = fit_transform(*sizes).scale
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.transform = zoom_about(
            self.transform,
            factor=factor,
            anchor=(event.position().x(), event.position().y()),
            image_size=sizes[0],
            view_size=sizes[1],
            minimum_scale=fit,
        )
        self._fit_mode = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._last_position = event.position()
        split = self.width() * self.split_ratio
        self._drag_mode = "split" if abs(event.position().x() - split) <= 14 else "pan"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "split":
            self.split_ratio = min(1.0, max(0.0, event.position().x() / max(1, self.width())))
            self.update()
        elif self._drag_mode == "pan" and self.transform is not None and self._sizes() is not None:
            delta = event.position() - self._last_position
            self.transform = pan_transform(
                self.transform,
                delta=(delta.x(), delta.y()),
                image_size=self._sizes()[0],
                view_size=self._sizes()[1],
            )
            self._fit_mode = False
            self._last_position = event.position()
            self.update()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._drag_mode = None


class PreviewWindow(QMainWindow):
    def __init__(self, source: Path, denoised: Path, cache_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.source = Path(source)
        self.denoised = Path(denoised)
        self.cache = PreviewCache(cache_root)
        self._request_id = 0
        self._thread: QThread | None = None
        self._worker: PreviewLoader | None = None
        self._pending_half_size: bool | None = None
        self._closed = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"对比预览 · {self.source.name}")
        self.resize(1280, 820)

        self.canvas = CompareCanvas()
        controls = QHBoxLayout()
        fit = QPushButton("适应窗口")
        fit.clicked.connect(self.canvas.fit_image)
        controls.addWidget(fit)
        one = QPushButton("100%")
        one.clicked.connect(lambda: self.load_preview(False))
        controls.addWidget(one)
        regenerate = QPushButton("重新生成")
        regenerate.clicked.connect(self.regenerate)
        controls.addWidget(regenerate)
        self.status = QLabel("正在生成离线预览…")
        controls.addWidget(self.status, 1)
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.canvas, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.load_preview(True)

    def load_preview(self, half_size: bool) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._pending_half_size = half_size
            return
        self._request_id += 1
        request_id = self._request_id
        self.status.setText("正在读取缓存或显影…")
        thread = QThread(self)
        worker = PreviewLoader(
            request_id, self.cache, self.source, self.denoised, half_size
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ready.connect(lambda rid, pair: self._preview_ready(rid, pair, half_size))
        worker.failed.connect(self._preview_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._load_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(int, object)
    def _preview_ready(self, request_id: int, pair: PreviewPair, half_size: bool) -> None:
        if self._closed or request_id != self._request_id:
            return
        self.canvas.set_pair(pair, one_to_one=not half_size)
        self.status.setText("拖动分割线对比 · 滚轮缩放 · 拖动图像平移")

    @Slot(int, str)
    def _preview_failed(self, request_id: int, message: str) -> None:
        if self._closed or request_id != self._request_id:
            return
        self.status.setText("预览失败")
        QMessageBox.warning(self, "无法生成预览", message)

    @Slot()
    def _load_finished(self) -> None:
        self._thread = None
        self._worker = None
        pending = self._pending_half_size
        self._pending_half_size = None
        if pending is not None and not self._closed:
            self.load_preview(pending)

    def regenerate(self) -> None:
        half_size = self.canvas.source_image is None or self.canvas.source_image.width() < 3000
        self.cache.invalidate(self.source, self.denoised, half_size=half_size)
        self.load_preview(half_size)

    def closeEvent(self, event) -> None:
        self._closed = True
        self._request_id += 1
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self.hide()
            self._thread.finished.connect(self.close)
            event.ignore()
            return
        event.accept()
