from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence


DEFAULT_PHASE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("decoding", 0.08),
    ("denoising", 0.55),
    ("postprocessing", 0.12),
    ("writing", 0.20),
    ("validating", 0.05),
)


class ProcessingCancelled(Exception):
    """A user-requested stop, distinct from a processing failure."""

    def __init__(self, message: str = "处理已取消") -> None:
        super().__init__(message)


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._event.set()
            return True

    def check(self) -> None:
        if self._event.is_set():
            raise ProcessingCancelled()


@dataclass(frozen=True)
class ProgressEvent:
    job_id: int | None
    phase: str
    completed: int
    total: int
    phase_progress: float
    overall: float
    timestamp: float


class ProgressTracker:
    def __init__(
        self,
        *,
        job_id: int | None,
        phase_weights: Sequence[tuple[str, float]] = DEFAULT_PHASE_WEIGHTS,
        clock: Callable[[], float] = time.monotonic,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        values = tuple(phase_weights)
        if not values:
            raise ValueError("阶段权重不能为空")
        names = [name for name, _weight in values]
        if len(set(names)) != len(names):
            raise ValueError("阶段名称不能重复")
        if any(not name for name in names):
            raise ValueError("阶段名称不能为空")
        if any(not math.isfinite(weight) or weight <= 0.0 for _name, weight in values):
            raise ValueError("阶段权重必须是有限正数")
        if not math.isclose(sum(weight for _name, weight in values), 1.0, abs_tol=1e-9):
            raise ValueError("阶段权重之和必须为 1")
        offset = 0.0
        self._phases: dict[str, tuple[float, float]] = {}
        for name, weight in values:
            self._phases[name] = (offset, weight)
            offset += weight
        self.job_id = job_id
        self.clock = clock
        self.on_progress = on_progress
        self._last_overall = 0.0
        self._last_timestamp = -math.inf
        self._lock = threading.Lock()

    def update(self, phase: str, completed: int, total: int) -> ProgressEvent:
        if phase not in self._phases:
            raise ValueError(f"未知处理阶段：{phase}")
        if total <= 0:
            raise ValueError("进度总量必须大于零")
        if completed < 0 or completed > total:
            raise ValueError("已完成数量超出进度范围")
        offset, weight = self._phases[phase]
        fraction = completed / total
        overall = offset + weight * fraction
        timestamp = float(self.clock())
        if not math.isfinite(timestamp):
            raise ValueError("进度时间戳无效")
        with self._lock:
            if overall + 1e-12 < self._last_overall:
                raise ValueError("总进度不能倒退")
            if timestamp < self._last_timestamp:
                raise ValueError("进度时间戳不能倒退")
            event = ProgressEvent(
                job_id=self.job_id,
                phase=phase,
                completed=completed,
                total=total,
                phase_progress=fraction,
                overall=overall,
                timestamp=timestamp,
            )
            self._last_overall = overall
            self._last_timestamp = timestamp
        if self.on_progress is not None:
            self.on_progress(event)
        return event


class TaskController:
    def __init__(
        self,
        *,
        cancellation: CancellationToken | None = None,
        progress_tracker: ProgressTracker | None = None,
    ) -> None:
        self.cancellation = cancellation or CancellationToken()
        self.progress_tracker = progress_tracker

    @property
    def is_cancelled(self) -> bool:
        return self.cancellation.is_cancelled

    def cancel(self) -> bool:
        return self.cancellation.cancel()

    def check(self) -> None:
        self.cancellation.check()

    def progress(self, phase: str, completed: int, total: int) -> ProgressEvent | None:
        self.check()
        if self.progress_tracker is None:
            return None
        return self.progress_tracker.update(phase, completed, total)
