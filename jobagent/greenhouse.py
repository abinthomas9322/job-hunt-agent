"""Search Irish jobs on employers' public Greenhouse job boards.

Many tech employers with Dublin offices publish every opening through
Greenhouse's public Job Board API — no key needed, and each posting includes
the full description. We fetch the configured boards once per session, keep
the postings located in Ireland, and rank them by keyword match.

Adzuna, the other source, does not cover Ireland, which is why this exists.
"""

import html
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from jobagent.config import Settings, get_settings
from jobagent.jobs import Job, JobSearchError

BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
IRELAND = re.compile(r"\b(ireland|dublin|cork|galway|limerick|waterford)\b", re.I)
TAG = re.compile(r"<[^>]+>")
WORD = re.compile(r"[a-z0-9+#]+")
# Words that don't help decide whether a job is relevant.
IGNORED = frozenset({"a", "an", "and", "the", "in", "of", "for", "to", "or", "job", "jobs", "role"})


def _plain_text(raw_html: str) -> str:
    text = TAG.sub(" ", html.unescape(html.unescape(raw_html)))
    return " ".join(text.split())


def _to_job(raw: dict[str, Any], board: str) -> Job:
    return Job(
        id=f"gh-{board}-{raw['id']}",
        title=raw.get("title", "").strip(),
        company=raw.get("company_name") or board.title(),
        location=(raw.get("location") or {}).get("name", "Unknown"),
        description=_plain_text(raw.get("content", ""))[:4000],
        url=raw.get("absolute_url", ""),
        created=raw.get("first_published") or raw.get("updated_at"),
    )


def _relevance(job: Job, terms: list[str]) -> int:
    """Title matches count triple; description matches count once."""
    title = set(WORD.findall(job.title.lower()))
    body = set(WORD.findall(job.description.lower()))
    return sum(3 * (t in title) + (t in body) for t in terms)


class GreenhouseClient:
    """Keyword search over the Irish postings of several Greenhouse boards."""

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self._client = client or httpx.Client(timeout=20.0)
        self._cache: list[Job] | None = None

    def _all_irish_jobs(self) -> list[Job]:
        if self._cache is None:
            jobs: list[Job] = []
            failures = 0
            for board in self.settings.greenhouse_boards:
                try:
                    response = self._client.get(
                        BOARD_URL.format(board=board), params={"content": "true"}
                    )
                    response.raise_for_status()
                except httpx.HTTPError:
                    failures += 1  # one broken board shouldn't sink the search
                    continue
                for raw in response.json().get("jobs", []):
                    job = _to_job(raw, board)
                    if IRELAND.search(job.location):
                        jobs.append(job)
            if failures == len(self.settings.greenhouse_boards):
                raise JobSearchError("could not reach any Greenhouse job board")
            self._cache = jobs
        return self._cache

    def search(
        self,
        what: str,
        where: str = "",
        *,
        max_days_old: int | None = None,
        limit: int = 10,
    ) -> list[Job]:
        """Return up to ``limit`` Irish jobs matching ``what``, most relevant first.

        ``where`` narrows the location (e.g. "Dublin"); ``max_days_old``
        drops postings first published longer ago than that.
        """
        terms = [t for t in WORD.findall(what.lower()) if t not in IGNORED]
        if not terms:
            raise ValueError("search keywords must not be empty")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")

        cutoff = datetime.now(UTC) - timedelta(days=max_days_old) if max_days_old else None
        scored = []
        for job in self._all_irish_jobs():
            if where.strip() and where.strip().lower() not in job.location.lower():
                continue
            if cutoff and job.created and job.created < cutoff:
                continue
            score = _relevance(job, terms)
            # Require a title hit so "engineer" doesn't return every job that
            # mentions engineers somewhere in a long description.
            if score >= 3:
                scored.append((score, job))
        oldest = datetime.min.replace(tzinfo=UTC)
        # Most relevant first; among equals, the newest posting first.
        scored.sort(key=lambda pair: (pair[0], pair[1].created or oldest), reverse=True)
        return [job for _, job in scored[:limit]]
