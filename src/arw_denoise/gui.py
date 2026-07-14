from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .config import AppPaths
from .dnglab import DngLabClient
from .gpu_probe import create_default_gpu_probe
from .gui_helpers import (
    format_duration,
    job_parameters,
    open_in_explorer,
    progress_eta,
    queue_progress,
)
from .jobs import Job, JobStore
from .metrics import ResourceMonitor
from .processor import AutoProcessingSettings, SmartRawProcessor
from .settings import AppSettings, SettingsStore, resolve_output_dir
from .task_control import ProcessingCancelled, ProgressEvent, ProgressTracker, TaskController


def run_gui() -> int:
    try:
        from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QProgressBar,
            QSlider,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("缺少 PySide6，请安装 arw-denoise[gui]") from exc

    class ProcessingWorker(QObject):
        updated = Signal()
        notice = Signal(str)
        finished = Signal()

        def __init__(self, store: JobStore):
            super().__init__()
            self.store = store
            self.pause_requested = threading.Event()
            self.cancel_requested = threading.Event()
            self._control_lock = threading.Lock()
            self._current_control: TaskController | None = None

        @Slot()
        def run(self) -> None:
            executor: ThreadPoolExecutor | None = None
            try:
                processor = SmartRawProcessor(dnglab=DngLabClient())
                jobs = self.store.list("queued")
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raw-prefetch")
                prefetched: Future | None = (
                    executor.submit(processor.decoder.decode, jobs[0].source_path) if jobs else None
                )
                for index, job in enumerate(jobs):
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
                        monitor = ResourceMonitor(interval_seconds=0.5).start()
                        self.store.transition(job.id, "decoding")
                        self.updated.emit()
                        started = time.monotonic()

                        def save_progress(event: ProgressEvent) -> None:
                            self.store.record_progress(
                                job.id,
                                phase=event.phase,
                                phase_progress=event.phase_progress,
                                overall_progress=event.overall,
                                elapsed_seconds=max(0.0, event.timestamp - started),
                            )
                            self.updated.emit()

                        control = TaskController(
                            progress_tracker=ProgressTracker(job.id, on_progress=save_progress)
                        )
                        with self._control_lock:
                            self._current_control = control
                        if self.cancel_requested.is_set():
                            control.cancel()

                        current_prefetch = prefetched
                        prefetched = None
                        decoded_frame = current_prefetch.result() if current_prefetch is not None else None
                        next_job = jobs[index + 1] if index + 1 < len(jobs) else None
                        prefetched = (
                            executor.submit(processor.decoder.decode, next_job.source_path)
                            if next_job is not None and not self.cancel_requested.is_set()
                            else None
                        )

                        def phase(name: str) -> None:
                            self.store.transition(job.id, name)
                            self.updated.emit()

                        mode = "gpu" if job.mode == "extreme" else job.mode
                        allowed = {
                            key: value
                            for key, value in job.parameters.items()
                            if key in {"strength", "chroma_noise", "detail_protection", "artifact_suppression"}
                        }
                        result = processor.process(
                            job.source_path,
                            job.output_path,
                            AutoProcessingSettings(mode=mode, **allowed),
                            on_phase=phase,
                            control=control,
                            decoded_frame=decoded_frame,
                        )
                        peaks = monitor.stop()
                        self.store.record_runtime(
                            job.id,
                            engine_id=result.engine.engine_id,
                            model_version=result.engine.model_version,
                            provider=result.engine.provider,
                            tile_size=result.stats.tile_size,
                            inference_seconds=result.stats.inference_seconds,
                            fallback_reason=result.fallback_reason,
                            peak_ram_mb=peaks.ram_mb,
                            peak_vram_mb=peaks.vram_mb,
                        )
                        self.store.transition(job.id, "validating")
                        self.store.transition(job.id, "completed")
                    except ProcessingCancelled:
                        current = self.store.get(job.id)
                        if current.state in {"decoding", "denoising", "writing", "validating"}:
                            self.store.transition(job.id, "cancelled")
                    except Exception as exc:
                        current = self.store.get(job.id)
                        if current.state in {"decoding", "denoising", "writing", "validating"}:
                            self.store.transition(job.id, "failed", str(exc))
                        self.notice.emit(f"{job.source_path.name}: {exc}")
                    finally:
                        if "monitor" in locals():
                            monitor.stop()
                            del monitor
                        with self._control_lock:
                            self._current_control = None
                    self.updated.emit()
            except Exception as exc:
                self.notice.emit(str(exc))
            finally:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                self.finished.emit()

        def pause(self) -> None:
            self.pause_requested.set()

        def resume(self) -> None:
            self.pause_requested.clear()

        def cancel(self) -> None:
            self.cancel_requested.set()
            self.pause_requested.clear()
            with self._control_lock:
                control = self._current_control
            if control is not None:
                control.cancel()

    class ProbeWorker(QObject):
        completed = Signal(object)
        failed = Signal(str)
        finished = Signal()

        @Slot()
        def run(self) -> None:
            try:
                self.completed.emit(create_default_gpu_probe().run(force=True))
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                self.finished.emit()

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ARW Denoise · 离线 RAW 降噪")
            self.resize(1260, 780)
            self.paths = AppPaths.default()
            self.paths.ensure()
            self.store = JobStore(self.paths.database)
            self.store.recover_interrupted()
            self.settings_store = SettingsStore(self.paths.settings)
            self.settings = self.settings_store.load()
            self.thread = None
            self.worker = None
            self.probe_thread = None
            self.probe_worker = None
            self._close_after_processing = False

            splitter = QSplitter()
            splitter.addWidget(self._sources_panel())
            splitter.addWidget(self._queue_panel())
            splitter.addWidget(self._settings_panel())
            splitter.setSizes([250, 650, 360])

            bottom = QHBoxLayout()
            self.summary = QLabel()
            bottom.addWidget(self.summary, 1)
            self.start_button = QPushButton("开始处理")
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

            progress = QVBoxLayout()
            self.progress_status = QLabel("尚未开始")
            progress.addWidget(self.progress_status)
            self.file_progress = QProgressBar()
            self.file_progress.setRange(0, 1000)
            self.file_progress.setFormat("当前文件 %p%")
            progress.addWidget(self.file_progress)
            self.queue_progress_bar = QProgressBar()
            self.queue_progress_bar.setRange(0, 1000)
            self.queue_progress_bar.setFormat("整体队列 %p%")
            progress.addWidget(self.queue_progress_bar)

            root = QVBoxLayout()
            root.addWidget(splitter, 1)
            root.addLayout(progress)
            root.addLayout(bottom)
            widget = QWidget()
            widget.setLayout(root)
            self.setCentralWidget(widget)
            self._load_settings_into_controls()
            self.refresh()
            QTimer.singleShot(250, self.probe_gpu)

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
            retry = QPushButton("重试失败任务")
            retry.clicked.connect(self.retry_failed)
            layout.addWidget(retry)
            retry_cancelled = QPushButton("重试已取消任务")
            retry_cancelled.clicked.connect(self.retry_cancelled)
            layout.addWidget(retry_cancelled)
            clear_completed = QPushButton("清理已完成记录")
            clear_completed.clicked.connect(self.clear_completed)
            layout.addWidget(clear_completed)
            open_output = QPushButton("打开导出目录")
            open_output.clicked.connect(self.open_output_folder)
            layout.addWidget(open_output)
            locate = QPushButton("定位选中 DNG")
            locate.clicked.connect(self.locate_selected_output)
            layout.addWidget(locate)
            layout.addStretch(1)
            return panel

        def _queue_panel(self):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            title = QLabel("批处理队列")
            title.setStyleSheet("font-size: 18px; font-weight: 600")
            layout.addWidget(title)
            self.queue = QListWidget()
            self.queue.itemDoubleClicked.connect(lambda _item: self.locate_selected_output())
            layout.addWidget(self.queue)
            return panel

        def _path_row(self, edit: QLineEdit, button_text: str, handler) -> QWidget:
            widget = QWidget()
            row = QHBoxLayout(widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(edit, 1)
            button = QPushButton(button_text)
            button.clicked.connect(handler)
            row.addWidget(button)
            return widget

        def _settings_panel(self):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            title = QLabel("设置")
            title.setStyleSheet("font-size: 18px; font-weight: 600")
            layout.addWidget(title)

            layout.addWidget(QLabel("处理引擎"))
            self.mode = QComboBox()
            self.mode.addItem("全自动（推荐）", "auto")
            self.mode.addItem("强制 NVIDIA GPU", "gpu")
            self.mode.addItem("CPU 兼容", "cpu")
            layout.addWidget(self.mode)

            self.gpu_status = QLabel("GPU：正在检测…")
            self.gpu_status.setWordWrap(True)
            layout.addWidget(self.gpu_status)
            self.gpu_probe_button = QPushButton("重新检测 GPU")
            self.gpu_probe_button.clicked.connect(self.probe_gpu)
            layout.addWidget(self.gpu_probe_button)

            layout.addWidget(QLabel("默认导入目录"))
            self.import_dir = QLineEdit()
            layout.addWidget(self._path_row(self.import_dir, "选择", self.choose_import_dir))
            layout.addWidget(QLabel("导出策略"))
            self.output_strategy = QComboBox()
            self.output_strategy.addItem("源目录 / DNG_Denoised", "source_subfolder")
            self.output_strategy.addItem("固定导出目录", "fixed")
            layout.addWidget(self.output_strategy)
            self.output_dir = QLineEdit()
            layout.addWidget(self._path_row(self.output_dir, "选择", self.choose_output_dir))

            self.advanced_toggle = QPushButton("高级设置 ▸")
            self.advanced_toggle.setCheckable(True)
            self.advanced_toggle.toggled.connect(self._toggle_advanced)
            layout.addWidget(self.advanced_toggle)
            self.advanced_panel = QWidget()
            advanced = QVBoxLayout(self.advanced_panel)
            advanced.setContentsMargins(0, 0, 0, 0)
            self.advanced_enabled = QCheckBox("启用手动高级参数（不勾选则保持全自动）")
            advanced.addWidget(self.advanced_enabled)
            self.sliders = {}
            specs = (
                ("strength", "降噪强度", 0, 200, 50),
                ("chroma_noise", "彩色噪点", 0, 100, 50),
                ("detail_protection", "细节保护", 0, 100, 82),
                ("artifact_suppression", "伪影抑制", 0, 100, 50),
            )
            for key, label, minimum, maximum, value in specs:
                caption = QLabel()
                slider = QSlider(Qt.Horizontal)
                slider.setRange(minimum, maximum)
                slider.setValue(value)
                slider.valueChanged.connect(lambda current, c=caption, name=label: c.setText(f"{name}：{current / 100:.2f}"))
                caption.setText(f"{label}：{value / 100:.2f}")
                advanced.addWidget(caption)
                advanced.addWidget(slider)
                self.sliders[key] = slider
            layout.addWidget(self.advanced_panel)
            note = QLabel("默认由 ISO 和 RAW 噪声估计自动调节。输出为保留 CFA、白平衡和高光余量的可编辑 DNG。")
            note.setWordWrap(True)
            note.setStyleSheet("color: #777; margin-top: 12px")
            layout.addWidget(note)
            layout.addStretch(1)
            return panel

        def _load_settings_into_controls(self) -> None:
            index = self.mode.findData(self.settings.engine_mode)
            self.mode.setCurrentIndex(max(0, index))
            self.import_dir.setText(self.settings.default_import_dir or "")
            self.output_dir.setText(self.settings.default_output_dir or "")
            index = self.output_strategy.findData(self.settings.output_strategy)
            self.output_strategy.setCurrentIndex(max(0, index))
            self.advanced_toggle.setChecked(self.settings.advanced_expanded)
            self.advanced_enabled.setChecked(self.settings.advanced_enabled)
            defaults = {"strength": 0.5, "chroma_noise": 0.5, "detail_protection": 0.82, "artifact_suppression": 0.5}
            for key, slider in self.sliders.items():
                value = getattr(self.settings, key)
                slider.setValue(round(100 * (defaults[key] if value is None else value)))
            self._toggle_advanced(self.settings.advanced_expanded)
            self.mode.currentIndexChanged.connect(self.save_settings)
            self.output_strategy.currentIndexChanged.connect(self._output_strategy_changed)
            self.import_dir.editingFinished.connect(self.save_settings)
            self.output_dir.editingFinished.connect(self.save_settings)
            self.advanced_enabled.toggled.connect(self.save_settings)
            for slider in self.sliders.values():
                slider.sliderReleased.connect(self.save_settings)

        def _current_settings(self) -> AppSettings:
            return AppSettings(
                default_import_dir=self.import_dir.text().strip() or None,
                default_output_dir=self.output_dir.text().strip() or None,
                output_strategy=str(self.output_strategy.currentData()),
                engine_mode=str(self.mode.currentData()),
                advanced_expanded=self.advanced_toggle.isChecked(),
                advanced_enabled=self.advanced_enabled.isChecked(),
                strength=self.sliders["strength"].value() / 100,
                chroma_noise=self.sliders["chroma_noise"].value() / 100,
                detail_protection=self.sliders["detail_protection"].value() / 100,
                artifact_suppression=self.sliders["artifact_suppression"].value() / 100,
            )

        @Slot()
        def save_settings(self, *_args) -> None:
            try:
                self.settings = self._current_settings()
                self.settings_store.save(self.settings)
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(f"无法保存设置：{exc}", 8000)

        def _toggle_advanced(self, expanded: bool) -> None:
            self.advanced_panel.setVisible(expanded)
            self.advanced_toggle.setText("高级设置 ▾" if expanded else "高级设置 ▸")
            if hasattr(self, "settings_store"):
                self.save_settings()

        def _output_strategy_changed(self, *_args) -> None:
            if self.output_strategy.currentData() == "fixed" and not self.output_dir.text().strip():
                self.choose_output_dir()
                if not self.output_dir.text().strip():
                    self.output_strategy.setCurrentIndex(self.output_strategy.findData("source_subfolder"))
            self.save_settings()

        def choose_import_dir(self) -> None:
            start = self.import_dir.text().strip() or str(Path.home())
            selected = QFileDialog.getExistingDirectory(self, "选择默认导入目录", start)
            if selected:
                self.import_dir.setText(selected)
                self.save_settings()

        def choose_output_dir(self) -> None:
            start = self.output_dir.text().strip() or self.import_dir.text().strip() or str(Path.home())
            selected = QFileDialog.getExistingDirectory(self, "选择固定导出目录", start)
            if selected:
                self.output_dir.setText(selected)
                self.output_strategy.setCurrentIndex(self.output_strategy.findData("fixed"))
                self.save_settings()

        def _selected_job(self) -> Job | None:
            item = self.queue.currentItem()
            if item is None:
                return None
            try:
                return self.store.get(int(item.data(Qt.UserRole)))
            except (KeyError, TypeError, ValueError):
                return None

        def _enqueue(self, paths: list[Path]) -> None:
            if not paths:
                return
            self.import_dir.setText(str(paths[0].parent.resolve()))
            self.save_settings()
            settings = self._current_settings()
            parameters = job_parameters(settings)
            for path in paths:
                self.store.add_with_available_output(
                    path,
                    resolve_output_dir(path, settings),
                    mode=settings.engine_mode,
                    parameters=parameters,
                )
            self.refresh()

        def add_files(self) -> None:
            start = self.import_dir.text().strip()
            files, _ = QFileDialog.getOpenFileNames(self, "选择 Sony ARW", start, "Sony RAW (*.ARW *.arw)")
            self._enqueue([Path(value) for value in files])

        def add_folder(self) -> None:
            start = self.import_dir.text().strip()
            folder = QFileDialog.getExistingDirectory(self, "选择包含 ARW 的文件夹", start)
            if not folder:
                return
            root = Path(folder)
            files = sorted({*root.glob("*.ARW"), *root.glob("*.arw")})
            if not files:
                QMessageBox.information(self, "未找到文件", "所选文件夹中没有 ARW 文件。")
                return
            self._enqueue(files)

        def open_output_folder(self) -> None:
            try:
                job = self._selected_job()
                if job is not None:
                    folder = job.output_path.parent
                else:
                    settings = self._current_settings()
                    source = settings.import_path or Path.home()
                    folder = resolve_output_dir(source / "placeholder.ARW", settings)
                folder.mkdir(parents=True, exist_ok=True)
                open_in_explorer(folder)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "无法打开", str(exc))

        def locate_selected_output(self) -> None:
            job = self._selected_job()
            if job is None:
                QMessageBox.information(self, "未选择任务", "请先选中一个已完成任务。")
                return
            try:
                open_in_explorer(job.output_path, select_file=True)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "无法定位", str(exc))

        def refresh(self) -> None:
            jobs = self.store.list()
            selected = self._selected_job()
            selected_id = selected.id if selected else None
            self.queue.clear()
            states = {"queued": "等待", "decoding": "解码", "denoising": "降噪", "writing": "写入", "validating": "验证", "completed": "完成", "failed": "失败", "cancelled": "已取消"}
            for job in jobs:
                detail = ""
                if job.provider:
                    detail = f"  · {job.provider} · {job.inference_seconds or 0:.2f}s"
                    if job.tile_size:
                        detail += f" · tile {job.tile_size}"
                    if job.peak_ram_mb is not None:
                        detail += f" · RAM {job.peak_ram_mb:.0f} MB"
                    if job.peak_vram_mb is not None:
                        detail += f" · VRAM {job.peak_vram_mb:.0f} MB"
                if job.fallback_reason:
                    detail += " · GPU 已回退 CPU"
                text = f"#{job.id}  [{states.get(job.state, job.state)}]  {job.source_path.name}  →  {job.output_path.name}{detail}"
                if job.error:
                    text += f"  ·  {job.error}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, job.id)
                item.setToolTip(job.error or job.fallback_reason or str(job.output_path))
                self.queue.addItem(item)
                if job.id == selected_id:
                    self.queue.setCurrentItem(item)
            queued = sum(job.state == "queued" for job in jobs)
            self.summary.setText(f"共 {len(jobs)} 张 · 等待 {queued} 张（队列无数量限制）")
            self.queue_progress_bar.setValue(round(queue_progress(jobs) * 1000))
            active = next(
                (job for job in jobs if job.state in {"decoding", "denoising", "writing", "validating"}),
                None,
            )
            if active is None:
                self.file_progress.setValue(0)
                self.progress_status.setText("尚未开始" if queued else "队列已结束")
            else:
                self.file_progress.setValue(round(active.overall_progress * 1000))
                phase_names = {
                    "decoding": "解码 RAW",
                    "denoising": "AI 降噪",
                    "postprocessing": "保护细节与高光",
                    "writing": "写入 DNG",
                    "validating": "校验 DNG",
                }
                eta = progress_eta(active.elapsed_seconds, active.overall_progress)
                eta_text = format_duration(eta) if eta is not None else "计算中"
                phase_name = phase_names.get(active.phase or active.state, active.phase or active.state)
                self.progress_status.setText(
                    f"{active.source_path.name} · {phase_name} · 已用 {format_duration(active.elapsed_seconds)} · 剩余约 {eta_text}"
                )
            running = self.thread is not None and self.thread.isRunning()
            self.start_button.setEnabled(not running and queued > 0)

        def retry_failed(self) -> None:
            for job in self.store.list("failed"):
                self.store.transition(job.id, "queued")
            self.refresh()

        def retry_cancelled(self) -> None:
            for job in self.store.list("cancelled"):
                self.store.transition(job.id, "queued")
            self.refresh()

        def clear_completed(self) -> None:
            removed = self.store.delete_completed()
            self.refresh()
            self.statusBar().showMessage(f"已清理 {removed} 条完成记录，DNG 文件已保留", 5000)

        def probe_gpu(self) -> None:
            if self.probe_thread is not None and self.probe_thread.isRunning():
                return
            self.gpu_probe_button.setEnabled(False)
            self.gpu_status.setText("GPU：正在执行真实 CUDA 推理自检…")
            self.probe_thread = QThread(self)
            self.probe_worker = ProbeWorker()
            self.probe_worker.moveToThread(self.probe_thread)
            self.probe_thread.started.connect(self.probe_worker.run)
            self.probe_worker.completed.connect(self._gpu_probe_completed)
            self.probe_worker.failed.connect(lambda message: self.gpu_status.setText(f"GPU：不可用 · {message}"))
            self.probe_worker.finished.connect(self.probe_thread.quit)
            self.probe_worker.finished.connect(self.probe_worker.deleteLater)
            self.probe_thread.finished.connect(self._probe_finished)
            self.probe_thread.finished.connect(self.probe_thread.deleteLater)
            self.probe_thread.start()

        @Slot(object)
        def _gpu_probe_completed(self, result) -> None:
            if result.success:
                self.gpu_status.setText(
                    f"GPU：{result.device_name} · {result.provider} · PMRID {result.model_version} · 自检 {result.inference_seconds:.2f}s"
                )
            else:
                self.gpu_status.setText(f"GPU：不可用 · {result.error}")

        @Slot()
        def _probe_finished(self) -> None:
            self.probe_worker = None
            self.probe_thread = None
            self.gpu_probe_button.setEnabled(True)

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
                self.statusBar().showMessage("正在立即取消当前照片并停止队列…")

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
                self.statusBar().showMessage("正在终止当前处理并清理临时文件，请稍候…")
                event.ignore()
                return
            if self.probe_thread is not None and self.probe_thread.isRunning():
                self.statusBar().showMessage("正在完成 GPU 自检，请稍候…")
                QTimer.singleShot(250, self.close)
                event.ignore()
                return
            self.save_settings()
            event.accept()

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return int(app.exec())
