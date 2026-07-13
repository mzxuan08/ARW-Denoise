from pathlib import Path

import pytest

from arw_denoise.jobs import JobStore


def test_job_state_machine_and_recovery(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.add(tmp_path / "a.ARW", tmp_path / "a_DN.dng")
    assert job.state == "queued"
    assert store.transition(job.id, "decoding").state == "decoding"
    with pytest.raises(ValueError):
        store.transition(job.id, "completed")
    assert store.recover_interrupted() == 1
    recovered = store.get(job.id)
    assert recovered.state == "queued"
    assert "中断" in (recovered.error or "")


def test_failed_job_can_retry(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.add(tmp_path / "a.ARW", tmp_path / "a.dng")
    store.transition(job.id, "decoding")
    store.transition(job.id, "failed", "bad raw")
    retried = store.transition(job.id, "queued")
    assert retried.state == "queued"
    assert retried.error is None

