"""Track job applications in a local SQLite file.

The agent saves a job here when you decide to apply, then updates its status as
you hear back — a small CRM for your job search.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

from jobagent.jobs import Job

Status = Literal["saved", "applied", "interview", "offer", "rejected"]
STATUSES: tuple[str, ...] = get_args(Status)


class Application(BaseModel):
    """A job you are tracking, and where it stands."""

    job_id: str
    title: str
    company: str
    url: str
    status: Status
    notes: str = ""
    updated_at: str


class Tracker:
    """SQLite-backed store of applications, keyed by the job's id."""

    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """create table if not exists applications(
                job_id text primary key,
                title text not null,
                company text not null,
                url text not null,
                status text not null,
                notes text not null default '',
                updated_at text not null
            )"""
        )
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    def save(self, job: Job, status: Status = "saved", notes: str = "") -> Application:
        """Start tracking ``job`` (or overwrite its entry if already tracked)."""
        self._check_status(status)
        self._conn.execute(
            """insert into applications(job_id, title, company, url, status, notes, updated_at)
               values (?, ?, ?, ?, ?, ?, ?)
               on conflict(job_id) do update set
                 status = excluded.status, notes = excluded.notes,
                 updated_at = excluded.updated_at""",
            (job.id, job.title, job.company, job.url, status, notes, self._now()),
        )
        self._conn.commit()
        return self.get(job.id)  # type: ignore[return-value]

    def update_status(self, job_id: str, status: Status, notes: str | None = None) -> Application:
        """Move a tracked application to a new status, optionally replacing its notes.

        Raises:
            KeyError: If the job is not being tracked.
        """
        self._check_status(status)
        current = self.get(job_id)
        if current is None:
            raise KeyError(f"no tracked application for job {job_id}")
        self._conn.execute(
            "update applications set status = ?, notes = ?, updated_at = ? where job_id = ?",
            (status, current.notes if notes is None else notes, self._now(), job_id),
        )
        self._conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Application | None:
        """Return one tracked application, or None."""
        row = self._conn.execute(
            "select * from applications where job_id = ?", (job_id,)
        ).fetchone()
        return Application(**dict(row)) if row else None

    def list(self, status: Status | None = None) -> list[Application]:
        """List tracked applications, newest update first, optionally by status."""
        if status is not None:
            self._check_status(status)
            rows = self._conn.execute(
                "select * from applications where status = ? order by updated_at desc, rowid desc",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "select * from applications order by updated_at desc, rowid desc"
            ).fetchall()
        return [Application(**dict(r)) for r in rows]

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _check_status(status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
