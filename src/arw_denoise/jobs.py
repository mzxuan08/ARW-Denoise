from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


STATES = {"queued", "decoding", "denoising", "writing", "validating", "completed", "failed", "cancelled"}
RUNNING_STATES = {"decoding", "denoising", "writing", "validating"}
ALLOWED_TRANSITIONS = {
    "queued": {"decoding", "cancelled"},
    "decoding": {"denoising", "failed", "cancelled"},
    "denoising": {"writing", "failed", "cancelled"},
    "writing": {"validating", "failed", "cancelled"},
    "validating": {"completed", "failed", "cancelled"},
    "failed": {"queued", "cancelled"},
    "cancelled": {"queued"},
    "completed": {"queued"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Job:
    id: int
    source_path: Path
    output_path: Path
    state: str
    mode: str
    parameters: dict
    error: str | None
    engine_id: str | None
    model_version: str | None
    provider: str | None
    tile_size: int | None
    inference_seconds: float | None
    fallback_reason: str | None
    phase: str | None
    phase_progress: float
    overall_progress: float
    elapsed_seconds: float
    peak_ram_mb: float | None
    peak_vram_mb: float | None
    cancelled_at: str | None
    created_at: str
    updated_at: str


class JobStore:
    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, id)")
            self._migrate_runtime_columns(db)
            self._repair_duplicate_outputs(db)
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_output_path ON jobs(output_path)")

    @staticmethod
    def _migrate_runtime_columns(db: sqlite3.Connection) -> None:
        existing = {str(row[1]) for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        columns = {
            "engine_id": "TEXT",
            "model_version": "TEXT",
            "provider": "TEXT",
            "tile_size": "INTEGER",
            "inference_seconds": "REAL",
            "fallback_reason": "TEXT",
            "phase": "TEXT",
            "phase_progress": "REAL NOT NULL DEFAULT 0",
            "overall_progress": "REAL NOT NULL DEFAULT 0",
            "elapsed_seconds": "REAL NOT NULL DEFAULT 0",
            "peak_ram_mb": "REAL",
            "peak_vram_mb": "REAL",
            "cancelled_at": "TEXT",
        }
        for name, sql_type in columns.items():
            if name not in existing:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")

    @staticmethod
    def _repair_duplicate_outputs(db: sqlite3.Connection) -> None:
        """Migrate pre-reservation databases without dropping queued work."""
        duplicates = db.execute(
            "SELECT output_path FROM jobs GROUP BY output_path HAVING COUNT(*) > 1"
        ).fetchall()
        if not duplicates:
            return
        reserved = {str(row[0]) for row in db.execute("SELECT output_path FROM jobs").fetchall()}
        for duplicate in duplicates:
            rows = db.execute(
                "SELECT id, output_path FROM jobs WHERE output_path = ? ORDER BY id", (duplicate[0],)
            ).fetchall()
            original = Path(rows[0]["output_path"])
            for row in rows[1:]:
                index = 2
                while True:
                    candidate = original.with_name(f"{original.stem}_{index}{original.suffix}")
                    if str(candidate) not in reserved and not candidate.exists():
                        break
                    index += 1
                db.execute("UPDATE jobs SET output_path = ?, updated_at = ? WHERE id = ?", (str(candidate), _now(), row["id"]))
                reserved.add(str(candidate))

    @staticmethod
    def _row(row: sqlite3.Row) -> Job:
        return Job(
            id=int(row["id"]),
            source_path=Path(row["source_path"]),
            output_path=Path(row["output_path"]),
            state=row["state"],
            mode=row["mode"],
            parameters=json.loads(row["parameters_json"]),
            error=row["error"],
            engine_id=row["engine_id"],
            model_version=row["model_version"],
            provider=row["provider"],
            tile_size=int(row["tile_size"]) if row["tile_size"] is not None else None,
            inference_seconds=(
                float(row["inference_seconds"]) if row["inference_seconds"] is not None else None
            ),
            fallback_reason=row["fallback_reason"],
            phase=row["phase"],
            phase_progress=float(row["phase_progress"] or 0.0),
            overall_progress=float(row["overall_progress"] or 0.0),
            elapsed_seconds=float(row["elapsed_seconds"] or 0.0),
            peak_ram_mb=float(row["peak_ram_mb"]) if row["peak_ram_mb"] is not None else None,
            peak_vram_mb=float(row["peak_vram_mb"]) if row["peak_vram_mb"] is not None else None,
            cancelled_at=row["cancelled_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def add(self, source: Path, output: Path, mode: str = "auto", parameters: dict | None = None) -> Job:
        if mode not in {"auto", "gpu", "cpu", "extreme"}:
            raise ValueError("未知处理模式")
        now = _now()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO jobs(source_path, output_path, state, mode, parameters_json, created_at, updated_at) VALUES(?, ?, 'queued', ?, ?, ?, ?)",
                (str(Path(source).resolve()), str(Path(output).resolve()), mode, json.dumps(parameters or {}, ensure_ascii=False), now, now),
            )
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (cursor.lastrowid,)).fetchone()
        assert row is not None
        return self._row(row)

    def add_with_available_output(
        self, source: Path, output_dir: Path, mode: str = "auto", parameters: dict | None = None
    ) -> Job:
        source = Path(source).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{source.stem}_DN"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            reserved = {Path(row[0]) for row in db.execute("SELECT output_path FROM jobs").fetchall()}
            index = 1
            while True:
                name = f"{stem}.dng" if index == 1 else f"{stem}_{index}.dng"
                candidate = output_dir / name
                if candidate not in reserved and not candidate.exists():
                    break
                index += 1
            now = _now()
            cursor = db.execute(
                "INSERT INTO jobs(source_path, output_path, state, mode, parameters_json, created_at, updated_at) VALUES(?, ?, 'queued', ?, ?, ?, ?)",
                (str(source), str(candidate), mode, json.dumps(parameters or {}, ensure_ascii=False), now, now),
            )
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (cursor.lastrowid,)).fetchone()
        assert row is not None
        return self._row(row)

    def get(self, job_id: int) -> Job:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row(row)

    def list(self, state: str | None = None) -> list[Job]:
        if state is not None and state not in STATES:
            raise ValueError("未知任务状态")
        with self._connect() as db:
            if state is None:
                rows = db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
            else:
                rows = db.execute("SELECT * FROM jobs WHERE state = ? ORDER BY id", (state,)).fetchall()
        return [self._row(row) for row in rows]

    def transition(self, job_id: int, new_state: str, error: str | None = None) -> Job:
        if new_state not in STATES:
            raise ValueError("未知任务状态")
        with self._connect() as db:
            row = db.execute("SELECT state, cancelled_at FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = row["state"]
            if new_state not in ALLOWED_TRANSITIONS[current]:
                raise ValueError(f"非法状态转换：{current} -> {new_state}")
            now = _now()
            cancelled_at = now if new_state == "cancelled" else (None if new_state == "queued" else row["cancelled_at"])
            db.execute(
                "UPDATE jobs SET state = ?, error = ?, cancelled_at = ?, updated_at = ? WHERE id = ?",
                (new_state, error, cancelled_at, now, job_id),
            )
            updated = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert updated is not None
        return self._row(updated)

    def record_runtime(
        self,
        job_id: int,
        *,
        engine_id: str,
        model_version: str | None,
        provider: str,
        tile_size: int | None,
        inference_seconds: float,
        fallback_reason: str | None,
        peak_ram_mb: float | None = None,
        peak_vram_mb: float | None = None,
    ) -> Job:
        if inference_seconds < 0:
            raise ValueError("推理耗时不能为负数")
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE jobs SET engine_id = ?, model_version = ?, provider = ?, tile_size = ?,
                inference_seconds = ?, fallback_reason = ?, peak_ram_mb = ?, peak_vram_mb = ?,
                updated_at = ? WHERE id = ?""",
                (
                    engine_id,
                    model_version,
                    provider,
                    tile_size,
                    inference_seconds,
                    fallback_reason,
                    peak_ram_mb,
                    peak_vram_mb,
                    _now(),
                    job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        return self._row(row)

    def record_progress(
        self,
        job_id: int,
        *,
        phase: str,
        phase_progress: float,
        overall_progress: float,
        elapsed_seconds: float,
    ) -> Job:
        if not phase:
            raise ValueError("处理阶段不能为空")
        if not 0.0 <= phase_progress <= 1.0 or not 0.0 <= overall_progress <= 1.0:
            raise ValueError("任务进度必须在 0–1 之间")
        if elapsed_seconds < 0:
            raise ValueError("已用时间不能为负数")
        with self._connect() as db:
            existing = db.execute(
                "SELECT overall_progress FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(job_id)
            if overall_progress + 1e-12 < float(existing["overall_progress"] or 0.0):
                raise ValueError("任务总进度不能倒退")
            db.execute(
                """UPDATE jobs SET phase = ?, phase_progress = ?, overall_progress = ?,
                elapsed_seconds = ?, updated_at = ? WHERE id = ?""",
                (phase, phase_progress, overall_progress, elapsed_seconds, _now(), job_id),
            )
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert row is not None
        return self._row(row)

    def recover_interrupted(self) -> int:
        with self._connect() as db:
            placeholders = ",".join("?" for _ in RUNNING_STATES)
            cursor = db.execute(
                f"UPDATE jobs SET state = 'queued', error = ?, updated_at = ? WHERE state IN ({placeholders})",
                ("上次运行中断，任务已恢复到队列", _now(), *sorted(RUNNING_STATES)),
            )
            return int(cursor.rowcount)
