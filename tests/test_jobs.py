from pathlib import Path
import sqlite3

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


def test_output_names_are_reserved_before_files_exist(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    first = store.add_with_available_output(tmp_path / "one" / "same.ARW", tmp_path / "out")
    second = store.add_with_available_output(tmp_path / "two" / "same.ARW", tmp_path / "out")
    assert first.output_path.name == "same_DN.dng"
    assert second.output_path.name == "same_DN_2.dng"


def test_legacy_duplicate_outputs_are_migrated(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            """CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source_path TEXT NOT NULL, output_path TEXT NOT NULL,
            state TEXT NOT NULL, mode TEXT NOT NULL, parameters_json TEXT NOT NULL, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        values = (str(tmp_path / "same.dng"), "queued", "cpu", "{}", "now", "now")
        db.execute("INSERT INTO jobs(source_path, output_path, state, mode, parameters_json, created_at, updated_at) VALUES('a.ARW', ?, ?, ?, ?, ?, ?)", values)
        db.execute("INSERT INTO jobs(source_path, output_path, state, mode, parameters_json, created_at, updated_at) VALUES('b.ARW', ?, ?, ?, ?, ?, ?)", values)
    jobs = JobStore(database).list()
    assert jobs[0].output_path.name == "same.dng"
    assert jobs[1].output_path.name == "same_2.dng"
