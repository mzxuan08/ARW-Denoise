from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class DurationSample:
    engine_id: str
    pixels: int
    seconds: float

    def validate(self) -> None:
        if not self.engine_id:
            raise ValueError("引擎 ID 不能为空")
        if self.pixels <= 0:
            raise ValueError("像素数必须大于零")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("耗时必须是有限正数")


class EtaEstimator:
    def __init__(self, *, max_samples: int = 8, minimum_samples: int = 2) -> None:
        if max_samples < minimum_samples or minimum_samples < 2:
            raise ValueError("ETA 样本数范围无效")
        self.max_samples = max_samples
        self.minimum_samples = minimum_samples
        self._samples: dict[str, deque[DurationSample]] = defaultdict(
            lambda: deque(maxlen=self.max_samples)
        )

    def add(self, sample: DurationSample) -> None:
        sample.validate()
        self._samples[sample.engine_id].append(sample)

    def sample_count(self, engine_id: str) -> int:
        return len(self._samples.get(engine_id, ()))

    def estimate(self, engine_id: str, pending_pixels: Iterable[int]) -> float | None:
        samples = tuple(self._samples.get(engine_id, ()))
        if len(samples) < self.minimum_samples:
            return None
        pixels = tuple(int(value) for value in pending_pixels)
        if not pixels or any(value <= 0 for value in pixels):
            return None
        rates = [sample.seconds / sample.pixels for sample in samples]
        center = statistics.median(rates)
        accepted = [rate for rate in rates if center / 4.0 <= rate <= center * 4.0]
        if len(accepted) < self.minimum_samples:
            return None
        seconds_per_pixel = statistics.median(accepted)
        return seconds_per_pixel * sum(pixels)
