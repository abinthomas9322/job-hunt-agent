"""Tests for the Greenhouse job-board source (HTTP mocked).

The postings below are hand-written test doubles in Greenhouse's documented
Job Board API shape — not real listings.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from jobagent.config import Settings
from jobagent.greenhouse import GreenhouseClient
from jobagent.jobs import JobSearchClient, JobSearchError
from jobagent.sources import build_job_source

RECENT = (datetime.now(UTC) - timedelta(days=3)).isoformat()
OLD = (datetime.now(UTC) - timedelta(days=90)).isoformat()


def _posting(
    pid: int, title: str, location: str, content: str = "", published: str = RECENT
) -> dict[str, Any]:
    return {
        "id": pid,
        "title": title,
        "company_name": "Acme",
        "location": {"name": location},
        "content": content,
        "absolute_url": f"https://example.test/{pid}",
        "first_published": published,
    }


BOARDS = {
    "acme": [
        _posting(
            1, "Graduate AI Engineer", "Dublin, Ireland", "&lt;p&gt;Python &amp;amp; LLMs&lt;/p&gt;"
        ),
        _posting(2, "AI Engineer", "London, England"),  # not in Ireland
        _posting(3, "Account Executive", "Dublin, Ireland", "work with engineers"),  # body-only hit
        _posting(4, "Machine Learning Engineer", "Cork, Ireland", published=OLD),
    ],
    "beta": [{"id": 9, "title": "Data Engineer", "location": {"name": "Remote - Ireland"}}],
}


def _client(
    boards: dict[str, list[dict[str, Any]]] | None = None, fail: set[str] | None = None
) -> tuple[GreenhouseClient, list[str]]:
    data = BOARDS if boards is None else boards
    calls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        board = request.url.path.split("/")[3]
        calls.append(board)
        if board in (fail or set()):
            return httpx.Response(500)
        return httpx.Response(200, json={"jobs": data.get(board, [])})

    settings = Settings(_env_file=None, greenhouse_boards=list(data) or ["x"])  # type: ignore[call-arg]
    return GreenhouseClient(settings, httpx.Client(transport=httpx.MockTransport(handle))), calls


def test_search_keeps_irish_title_matches_ranked_and_cleans_html() -> None:
    client, _ = _client()
    jobs = client.search("ai engineer")
    # Equal relevance for 4 and 9 ("engineer" in the title): the dated posting
    # ranks above the undated one. The body-only match (3) and London (2) are out.
    assert [j.id for j in jobs] == ["gh-acme-1", "gh-acme-4", "gh-beta-9"]
    first = jobs[0]
    assert first.description == "Python & LLMs"
    assert first.company == "Acme" and first.location == "Dublin, Ireland"
    assert jobs[2].company == "Beta"  # falls back to the board name


def test_where_and_max_days_old_filter() -> None:
    client, _ = _client()
    assert [j.id for j in client.search("engineer", "cork")] == ["gh-acme-4"]
    assert "gh-acme-4" not in [j.id for j in client.search("engineer", max_days_old=30)]


def test_boards_are_fetched_once_per_session() -> None:
    client, calls = _client()
    client.search("engineer")
    client.search("analyst")
    assert calls == ["acme", "beta"]


def test_one_failing_board_is_skipped_but_all_failing_raises() -> None:
    client, _ = _client(fail={"beta"})
    assert client.search("engineer")
    client, _ = _client(fail={"acme", "beta"})
    with pytest.raises(JobSearchError):
        client.search("engineer")


@pytest.mark.parametrize(("what", "limit"), [("the and", 5), ("ai", 0), ("ai", 51)])
def test_invalid_arguments(what: str, limit: int) -> None:
    client, _ = _client()
    with pytest.raises(ValueError):
        client.search(what, limit=limit)


def test_default_http_client_and_source_factory() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert isinstance(build_job_source(settings), GreenhouseClient)
    adzuna = Settings(_env_file=None, job_source="adzuna")  # type: ignore[call-arg]
    assert isinstance(build_job_source(adzuna), JobSearchClient)
