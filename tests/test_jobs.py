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


def test_runtime_statistics_are_persisted(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.add(tmp_path / "a.ARW", tmp_path / "a.dng", mode="gpu")
    updated = store.record_runtime(
        job.id,
        engine_id="onnx-pmrid",
        model_version="1.0.0",
        provider="CUDAExecutionProvider",
        tile_size=1024,
        inference_seconds=3.75,
        fallback_reason=None,
    )
    assert updated.engine_id == "onnx-pmrid"
    assert updated.model_version == "1.0.0"
    assert updated.provider == "CUDAExecutionProvider"
    assert updated.tile_size == 1024
    assert updated.inference_seconds == pytest.approx(3.75)


def test_legacy_database_is_migrated_with_empty_runtime_statistics(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            """CREATE TABLE jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source_path TEXT NOT NULL, output_path TEXT NOT NULL,
            state TEXT NOT NULL, mode TEXT NOT NULL, parameters_json TEXT NOT NULL, error TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        db.execute(
            "INSERT INTO jobs(source_path, output_path, state, mode, parameters_json, created_at, updated_at) "
            "VALUES('a.ARW', 'a.dng', 'queued', 'cpu', '{}', 'now', 'now')"
        )
    job = JobStore(database).list()[0]
    assert job.engine_id is None
    assert job.provider is None
    assert job.inference_seconds is None
    assert job.phase is None
    assert job.overall_progress == 0.0
    assert job.cancelled_at is None


def test_progress_and_resource_peaks_are_persisted(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.add(tmp_path / "a.ARW", tmp_path / "a.dng")
    updated = store.record_progress(
        job.id,
        phase="denoising",
        phase_progress=0.5,
        overall_progress=0.35,
        elapsed_seconds=4.2,
    )
    assert updated.phase == "denoising"
    assert updated.phase_progress == pytest.approx(0.5)
    assert updated.overall_progress == pytest.approx(0.35)
    assert updated.elapsed_seconds == pytest.approx(4.2)
    measured = store.record_runtime(
        job.id,
        engine_id="onnx-pmrid",
        model_version="1.0.0",
        provider="CUDAExecutionProvider",
        tile_size=1024,
        inference_seconds=3.1,
        fallback_reason=None,
        peak_ram_mb=900.0,
        peak_vram_mb=2300.0,
    )
    assert measured.peak_ram_mb == pytest.approx(900.0)
    assert measured.peak_vram_mb == pytest.approx(2300.0)


def test_user_cancel_is_timestamped_and_retry_clears_cancel_marker(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.add(tmp_path / "a.ARW", tmp_path / "a.dng")
    cancelled = store.transition(job.id, "cancelled")
    assert cancelled.cancelled_at is not None
    retried = store.transition(job.id, "queued")
    assert retried.cancelled_at is None


def test_retry_resets_old_progress(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.add(tmp_path / "a.ARW", tmp_path / "a.dng")
    store.record_progress(
        job.id, phase="denoising", phase_progress=0.5, overall_progress=0.4, elapsed_seconds=2
    )
    store.transition(job.id, "cancelled")
    retried = store.transition(job.id, "queued")
    assert retried.phase is None
    assert retried.overall_progress == 0
    assert retried.elapsed_seconds == 0


def test_delete_completed_only_removes_history_records(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    output = tmp_path / "finished.dng"
    output.write_bytes(b"keep me")
    completed = store.add(tmp_path / "a.ARW", output)
    store.transition(completed.id, "decoding")
    store.transition(completed.id, "denoising")
    store.transition(completed.id, "writing")
    store.transition(completed.id, "validating")
    store.transition(completed.id, "completed")
    queued = store.add(tmp_path / "b.ARW", tmp_path / "queued.dng")

    assert store.delete_completed() == 1
    assert output.read_bytes() == b"keep me"
    assert [job.id for job in store.list()] == [queued.id]
