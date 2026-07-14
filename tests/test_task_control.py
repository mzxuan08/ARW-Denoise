from __future__ import annotations

from dataclasses import asdict

import pytest

from arw_denoise.task_control import (
    CancellationToken,
    ProcessingCancelled,
    ProgressTracker,
)


def test_cancellation_is_idempotent_and_raises_dedicated_result() -> None:
    token = CancellationToken()
    token.check()
    assert token.cancel() is True
    assert token.cancel() is False
    assert token.is_cancelled is True
    with pytest.raises(ProcessingCancelled, match="已取消"):
        token.check()


def test_progress_is_weighted_monotonic_and_serializable() -> None:
    times = iter((10.0, 11.0, 12.0))
    events = []
    tracker = ProgressTracker(
        job_id=7,
        phase_weights=(("decoding", 1.0),),
        clock=lambda: next(times),
        on_progress=events.append,
    )
    first = tracker.update("decoding", 0, 2)
    second = tracker.update("decoding", 1, 2)
    third = tracker.update("decoding", 2, 2)
    assert [event.overall for event in events] == [0.0, 0.5, 1.0]
    assert [event.timestamp for event in events] == [10.0, 11.0, 12.0]
    assert third.job_id == 7
    assert asdict(first)["phase"] == "decoding"
    assert second.completed == 1


def test_progress_rejects_unknown_invalid_or_backward_updates() -> None:
    tracker = ProgressTracker(
        job_id=None,
        phase_weights=(("decode", 0.25), ("write", 0.75)),
    )
    with pytest.raises(ValueError, match="未知"):
        tracker.update("other", 0, 1)
    with pytest.raises(ValueError, match="总量"):
        tracker.update("decode", 0, 0)
    with pytest.raises(ValueError, match="范围"):
        tracker.update("decode", 2, 1)
    tracker.update("write", 1, 2)
    with pytest.raises(ValueError, match="倒退"):
        tracker.update("decode", 1, 1)


def test_phase_weights_must_be_unique_positive_and_sum_to_one() -> None:
    with pytest.raises(ValueError, match="重复"):
        ProgressTracker(job_id=1, phase_weights=(("a", 0.5), ("a", 0.5)))
    with pytest.raises(ValueError, match="正数"):
        ProgressTracker(job_id=1, phase_weights=(("a", 1.0), ("b", 0.0)))
    with pytest.raises(ValueError, match="1"):
        ProgressTracker(job_id=1, phase_weights=(("a", 0.2), ("b", 0.2)))
