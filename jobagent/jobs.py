"""Search real job listings through the Adzuna API.

Adzuna aggregates listings from job boards and company sites and offers a free
API key. We map its JSON into a small ``Job`` model so the rest of the agent
never depends on Adzuna's field names.
"""

from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

from jobagent.config import Settings, get_settings


class Job(BaseModel):
    """One job listing, normalised from the provider's response."""

    id: str
    title: str
    company: str
    location: str
    description: str  # Adzuna returns a snippet (~500 chars), not the full ad
    url: str
    created: datetime | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    contract_time: str | None = None  # "full_time" / "part_time" when known


class JobSearchError(RuntimeError):
    """Raised when the job provider can't be reached or rejects the request."""


def _to_job(raw: dict[str, Any]) -> Job:
    return Job(
        id=str(raw["id"]),
        title=raw.get("title", "").strip(),
        company=(raw.get("company") or {}).get("display_name", "Unknown"),
        location=(raw.get("location") or {}).get("display_name", "Unknown"),
        description=raw.get("description", "").strip(),
        url=raw.get("redirect_url", ""),
        created=raw.get("created"),
        salary_min=raw.get("salary_min"),
        salary_max=raw.get("salary_max"),
        contract_time=raw.get("contract_time"),
    )


class JobSearchClient:
    """Thin client for Adzuna's search endpoint.

    Pass ``client`` to inject an ``httpx.Client`` (tests use a mock transport);
    otherwise one is created per client instance.
    """

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self._client = client or httpx.Client(timeout=15.0)

    def search(
        self,
        what: str,
        where: str = "",
        *,
        max_days_old: int | None = None,
        limit: int = 10,
    ) -> list[Job]:
        """Return up to ``limit`` jobs matching keywords ``what`` near ``where``.

        Raises:
            JobSearchError: If credentials are missing or the API call fails.
            ValueError: If ``what`` is empty or ``limit`` is out of range.
        """
        if not what.strip():
            raise ValueError("search keywords must not be empty")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        s = self.settings
        if not (s.adzuna_app_id and s.adzuna_app_key):
            raise JobSearchError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in .env")

        params: dict[str, str | int] = {
            "app_id": s.adzuna_app_id,
            "app_key": s.adzuna_app_key,
            "what": what,
            "results_per_page": limit,
            "sort_by": "date",
        }
        if where.strip():
            params["where"] = where
        if max_days_old is not None:
            params["max_days_old"] = max_days_old

        url = f"{s.adzuna_base_url}/{s.adzuna_country}/search/1"
        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise JobSearchError(f"job search failed: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise JobSearchError(f"job search failed: {exc.__class__.__name__}") from exc

        return [_to_job(raw) for raw in response.json().get("results", [])]
