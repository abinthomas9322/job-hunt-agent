"""Tests for the Adzuna job search client.

The HTTP API is an external boundary, so these tests use ``httpx.MockTransport``.
The response below is a hand-written test double in Adzuna's documented shape —
not real listings.
"""

import httpx
import pytest

from jobagent.config import Settings
from jobagent.jobs import JobSearchClient, JobSearchError

ADZUNA_RESPONSE = {
    "count": 2,
    "results": [
        {
            "id": "4242",
            "title": " Graduate AI Engineer ",
            "company": {"display_name": "Example Ltd"},
            "location": {"display_name": "Dublin, Ireland"},
            "description": "Build LLM applications with Python and RAG.",
            "redirect_url": "https://example.test/jobs/4242",
            "created": "2026-09-20T10:00:00Z",
            "salary_min": 40000,
            "salary_max": 50000,
            "contract_time": "full_time",
        },
        {"id": 7, "title": "Data Analyst"},  # sparse listing: optional fields missing
    ],
}


def _settings(**overrides: str) -> Settings:
    values = {"adzuna_app_id": "id", "adzuna_app_key": "key", **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


def _client(handler: httpx.MockTransport, **overrides: str) -> JobSearchClient:
    return JobSearchClient(_settings(**overrides), client=httpx.Client(transport=handler))


def test_search_maps_results_and_sends_expected_query() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ADZUNA_RESPONSE)

    jobs = _client(httpx.MockTransport(handle)).search(
        "ai engineer", "Dublin", max_days_old=14, limit=5
    )

    assert [j.title for j in jobs] == ["Graduate AI Engineer", "Data Analyst"]
    first = jobs[0]
    assert first.company == "Example Ltd"
    assert first.location == "Dublin, Ireland"
    assert first.salary_min == 40000
    assert first.created is not None
    assert jobs[1].company == "Unknown" and jobs[1].id == "7"

    params = seen[0].url.params
    assert seen[0].url.path == "/v1/api/jobs/gb/search/1"  # Adzuna has no Ireland
    assert params["what"] == "ai engineer"
    assert params["where"] == "Dublin"
    assert params["max_days_old"] == "14"
    assert params["results_per_page"] == "5"


def test_search_omits_optional_filters_when_not_given() -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    assert _client(httpx.MockTransport(handle)).search("python") == []
    assert "where" not in seen[0].url.params
    assert "max_days_old" not in seen[0].url.params


def test_search_requires_credentials() -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200)), adzuna_app_key="")
    with pytest.raises(JobSearchError, match="ADZUNA_APP_ID"):
        client.search("python")


@pytest.mark.parametrize(("what", "limit"), [("  ", 10), ("python", 0), ("python", 51)])
def test_search_validates_arguments(what: str, limit: int) -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ValueError):
        client.search(what, limit=limit)


def test_http_error_status_becomes_job_search_error() -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(JobSearchError, match="HTTP 401"):
        client.search("python")


def test_network_error_becomes_job_search_error() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(JobSearchError, match="ConnectError"):
        _client(httpx.MockTransport(handle)).search("python")


def test_default_client_is_created() -> None:
    assert JobSearchClient(_settings())._client is not None
