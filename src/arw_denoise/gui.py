from __future__ import annotations

from pathlib import Path
import threading
import time

from .config import AppPaths
from .jobs import JobStore
from .dnglab import DngLabClient
from .processor import CpuRawProcessor, ProcessingSettings


def run_gui() -> int:
    try:
        from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow,
            QMessageBox, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget, QComboBox,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 PySide6；请安装 arw-denoise[gui]") from exc

    class ProcessingWorker(QObject):
        updated = Signal()
        notice = Signal(str)
        finished = Signal()

        def __init__(self, store: JobStore):
            super().__init__()
            self.store = store
            self.pause_requested = threading.Event()
            self.cancel_requested = threading.Event()

        @Slot()
        def run(self) -> None:
            try:
                processor = CpuRawProcessor(dnglab=DngLabClient())
                for job in self.store.list("queued"):
                    if self.cancel_requested.is_set():
                        self.store.transition(job.id, "cancelled")
                        self.updated.emit()
                        continue
                    while self.pause_requested.is_set() and not self.cancel_requested.is_set():
                        time.sleep(0.1)
                    if self.cancel_requested.is_set():
                        self.store.transition(job.id, "cancelled")
                        self.updated.emit()
                        continue
                    try:
                        self.store.transition(job.id, "decoding")
                        self.updated.emit()

                        def phase(name: str) -> None:
                            self.store.transition(job.id, name)
                            self.updated.emit()

                        strength = float(job.parameters.get("strength", 1.0))
                        processor.process(job.source_path, job.output_path, ProcessingSettings(strength=strength), phase)
                        self.store.transition(job.id, "validating")
                        self.store.transition(job.id, "completed")
                    except Exception as exc:
                        current = self.store.get(job.id)
                        if current.state in {"decoding", "denoising", "writing", "validating"}:
                            self.store.transition(job.id, "failed", str(exc))
                        self.notice.emit(f"{job.source_path.name}: {exc}")
                    self.updated.emit()
            except Exception as exc:
                self.notice.emit(str(exc))
            finally:
                self.finished.emit()

        def pause(self) -> None:
            self.pause_requested.set()

        def resume(self) -> None:
            self.pause_requested.clear()

        def cancel(self) -> None:
            self.cancel_requested.set()
            self.pause_requested.clear()

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ARW Denoise")
            self.resize(1180, 720)
            paths = AppPaths.default()
            paths.ensure()
            self.store = JobStore(paths.database)
            self.store.recover_interrupted()
            self.thread = None
            self.worker = None
            self._close_after_processing = False

            splitter = QSplitter()
            splitter.addWidget(self._sources_panel())
            splitter.addWidget(self._queue_panel())
            splitter.addWidget(self._settings_panel())
            splitter.setSizes([230, 650, 300])

            bottom = QHBoxLayout()
            self.summary = QLabel()
            bottom.addWidget(self.summary, 1)
            self.start_button = QPushButton("开始")
            self.start_button.clicked.connect(self.start_processing)
            bottom.addWidget(self.start_button)
            self.pause_button = QPushButton("暂停")
            self.pause_button.clicked.connect(self.pause_processing)
            self.pause_button.setEnabled(False)
            bottom.addWidget(self.pause_button)
            self.resume_button = QPushButton("继续")
            self.resume_button.clicked.connect(self.resume_processing)
            self.resume_button.setEnabled(False)
            bottom.addWidget(self.resume_button)
            self.cancel_button = QPushButton("取消")
            self.cancel_button.clicked.connect(self.cancel_processing)
            self.cancel_button.setEnabled(False)
            bottom.addWidget(self.cancel_button)

            root = QVBoxLayout()
            root.addWidget(splitter, 1)
            root.addLayout(bottom)
            widget = QWidget()
            widget.setLayout(root)
            self.setCentralWidget(widget)
            self.refresh()

        def _sources_panel(self):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            title = QLabel("来源与任务")
            title.setStyleSheet("font-size: 18px; font-weight: 600")
            layout.addWidget(title)
            add_files = QPushButton("添加 ARW 文件")
            add_files.clicked.connect(self.add_files)
            layout.addWidget(add_files)
            add_folder = QPushButton("添加文件夹")
            add_folder.clicked.connect(self.add_folder)
            layout.addWidget(add_folder)
            layout.addStretch(1)
            return panel

        def _queue_panel(self):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            title = QLabel("批处理队列")
            title.setStyleSheet("font-size: 18px; font-weight: 600")
            layout.addWidget(title)
            self.queue = QListWidget()
            layout.addWidget(self.queue)
            return panel

        def _settings_panel(self):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            title = QLabel("降噪设置")
            title.setStyleSheet("font-size: 18px; font-weight: 600")
            layout.addWidget(title)
            self.mode = QComboBox()
            self.mode.addItems(["CPU 兼容（当前可用）"])
            layout.addWidget(self.mode)
            self.sliders = {}
            for index, label in enumerate(("降噪强度", "彩色噪点（GPU 版本）", "细节保护（GPU 版本）", "伪影抑制（GPU 版本）")):
                layout.addWidget(QLabel(label))
                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 100)
                slider.setValue(50)
                slider.setEnabled(index == 0)
                self.sliders[label] = slider
                layout.addWidget(slider)
            note = QLabel("当前版本使用保守的 CPU Bayer 小波基线并输出无压缩 CFA DNG；正式使用前仍需完成 A7C II 的 Adobe/像素蛋糕兼容测试。")
            note.setWordWrap(True)
            note.setStyleSheet("color: #777; margin-top: 16px")
            layout.addWidget(note)
            layout.addStretch(1)
            return panel

        def _enqueue(self, paths: list[Path]) -> None:
            if not paths:
                return
            output_dir = paths[0].parent / "DNG_Denoised"
            strength = self.sliders["降噪强度"].value() / 50.0
            for path in paths:
                self.store.add_with_available_output(path, output_dir, mode="cpu", parameters={"strength": strength})
            self.refresh()

        def add_files(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(self, "选择 Sony ARW", "", "Sony RAW (*.ARW *.arw)")
            self._enqueue([Path(value) for value in files])

        def add_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "选择包含 ARW 的文件夹")
            if not folder:
                return
            files = sorted(Path(folder).glob("*.ARW")) + sorted(Path(folder).glob("*.arw"))
            if not files:
                QMessageBox.information(self, "未找到文件", "所选文件夹中没有 ARW 文件。")
                return
            self._enqueue(files)

        def refresh(self) -> None:
            jobs = self.store.list()
            self.queue.clear()
            for job in jobs:
                self.queue.addItem(f"#{job.id}  [{job.state}]  {job.source_path.name}  →  {job.output_path.name}")
            self.summary.setText(f"共 {len(jobs)} 张 · 等待 {sum(job.state == 'queued' for job in jobs)} 张")
            running = self.thread is not None and self.thread.isRunning()
            self.start_button.setEnabled(not running and any(job.state == "queued" for job in jobs))

        def start_processing(self) -> None:
            if self.thread is not None and self.thread.isRunning():
                return
            self.thread = QThread(self)
            self.worker = ProcessingWorker(self.store)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.updated.connect(self.refresh)
            self.worker.notice.connect(lambda message: self.statusBar().showMessage(message, 12000))
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.processing_finished)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            self.statusBar().showMessage("正在处理队列…")

        def pause_processing(self) -> None:
            if self.worker:
                self.worker.pause()
                self.pause_button.setEnabled(False)
                self.resume_button.setEnabled(True)
                self.statusBar().showMessage("将在当前照片完成后暂停")

        def resume_processing(self) -> None:
            if self.worker:
                self.worker.resume()
                self.pause_button.setEnabled(True)
                self.resume_button.setEnabled(False)
                self.statusBar().showMessage("继续处理")

        def cancel_processing(self) -> None:
            if self.worker:
                self.worker.cancel()
                self.cancel_button.setEnabled(False)
                self.statusBar().showMessage("将在当前照片完成后取消剩余任务")

        @Slot()
        def processing_finished(self) -> None:
            self.worker = None
            self.thread = None
            self.pause_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("队列处理结束", 5000)
            self.refresh()
            if self._close_after_processing:
                QTimer.singleShot(0, self.close)

        def closeEvent(self, event) -> None:
            if self.thread is not None and self.thread.isRunning():
                self._close_after_processing = True
                if self.worker:
                    self.worker.cancel()
                self.statusBar().showMessage("正在安全结束当前照片，请稍候…")
                event.ignore()
                return
            event.accept()

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return int(app.exec())
