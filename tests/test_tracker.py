"""Tests for the SQLite application tracker (real in-memory / temp-file DB)."""

from pathlib import Path

import pytest

from jobagent.config import get_settings
from jobagent.jobs import Job
from jobagent.tracker import Tracker


def _job(job_id: str = "1", title: str = "AI Engineer") -> Job:
    return Job(
        id=job_id,
        title=title,
        company="Example Ltd",
        location="Dublin",
        description="",
        url=f"https://example.test/{job_id}",
    )


def test_save_then_get() -> None:
    tracker = Tracker()
    app = tracker.save(_job(), notes="referral from alumni")
    assert app.status == "saved"
    assert app.notes == "referral from alumni"
    assert tracker.get("1") == app
    assert tracker.get("missing") is None


def test_saving_again_updates_instead_of_duplicating() -> None:
    tracker = Tracker()
    tracker.save(_job())
    tracker.save(_job(), status="applied")
    assert [a.status for a in tracker.list()] == ["applied"]


def test_update_status_keeps_or_replaces_notes() -> None:
    tracker = Tracker()
    tracker.save(_job(), notes="first note")
    assert tracker.update_status("1", "interview").notes == "first note"
    assert tracker.update_status("1", "offer", notes="accepted").notes == "accepted"


def test_update_status_unknown_job_raises() -> None:
    with pytest.raises(KeyError):
        Tracker().update_status("nope", "applied")


def test_list_filters_by_status_newest_first() -> None:
    tracker = Tracker()
    tracker.save(_job("1", "First"))
    tracker.save(_job("2", "Second"), status="applied")
    tracker.save(_job("3", "Third"))
    assert [a.title for a in tracker.list()] == ["Third", "Second", "First"]
    assert [a.title for a in tracker.list("saved")] == ["Third", "First"]


@pytest.mark.parametrize("call", ["save", "update", "list"])
def test_invalid_status_is_rejected(call: str) -> None:
    tracker = Tracker()
    tracker.save(_job())
    with pytest.raises(ValueError, match="status must be one of"):
        if call == "save":
            tracker.save(_job(), status="ghosted")  # type: ignore[arg-type]
        elif call == "update":
            tracker.update_status("1", "ghosted")  # type: ignore[arg-type]
        else:
            tracker.list("ghosted")  # type: ignore[arg-type]


def test_data_persists_to_disk(tmp_path: Path) -> None:
    path = str(tmp_path / "nested" / "apps.db")
    first = Tracker(path)
    first.save(_job())
    first.close()
    assert Tracker(path).get("1") is not None


def test_settings_default_db_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.chdir(Path(__file__).parent)  # no .env here
    assert get_settings().db_path == "data/applications.db"
