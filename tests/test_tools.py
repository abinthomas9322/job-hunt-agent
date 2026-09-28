"""Tests for the agent's tools, called directly (no LLM involved)."""

import json

from langchain_core.tools import BaseTool

from jobagent.tools import build_tools
from jobagent.tracker import Tracker
from tests.fakes import FakeSearch, job


def _tools(search: FakeSearch | None = None) -> tuple[dict[str, BaseTool], Tracker]:
    tracker = Tracker()
    tools = build_tools(search or FakeSearch(), tracker)  # type: ignore[arg-type]
    return {t.name: t for t in tools}, tracker


def test_search_returns_json_summaries_and_caps_limit() -> None:
    search = FakeSearch()
    tools, _ = _tools(search)
    out = json.loads(tools["search_jobs"].invoke({"what": "ai engineer", "limit": 99}))
    assert out[0]["title"] == "Graduate AI Engineer"
    assert out[0]["salary"] == "40000.0-? EUR"
    assert out[0]["posted"] == "2026-09-20"
    assert search.queries[0]["limit"] == 20


def test_search_reports_empty_results_and_errors_as_text() -> None:
    tools, _ = _tools(FakeSearch(jobs=[]))
    assert tools["search_jobs"].invoke({"what": "x"}).startswith("No jobs found")
    tools, _ = _tools(FakeSearch(error="HTTP 401"))
    assert tools["search_jobs"].invoke({"what": "x"}) == "Error: HTTP 401"


def test_summary_handles_missing_salary_and_date() -> None:
    bare = job()
    bare.salary_min = None
    bare.created = None
    tools, _ = _tools(FakeSearch(jobs=[bare]))
    out = json.loads(tools["search_jobs"].invoke({"what": "x"}))
    assert out[0]["salary"] is None and out[0]["posted"] is None


def test_save_requires_a_job_seen_in_a_search() -> None:
    tools, tracker = _tools()
    assert tools["save_application"].invoke({"job_id": "101"}).startswith("Error: unknown")
    tools["search_jobs"].invoke({"what": "ai"})
    reply = tools["save_application"].invoke({"job_id": "101", "notes": "top pick"})
    assert reply == "Saved: Graduate AI Engineer at Example Ltd (status: saved)."
    assert tracker.get("101") is not None


def test_update_and_list_applications() -> None:
    tools, _ = _tools()
    assert tools["list_applications"].invoke({}) == "No tracked applications yet."
    assert (
        tools["update_application"]
        .invoke({"job_id": "101", "status": "applied"})
        .startswith("Error:")
    )
    tools["search_jobs"].invoke({"what": "ai"})
    tools["save_application"].invoke({"job_id": "101"})
    reply = tools["update_application"].invoke({"job_id": "101", "status": "interview"})
    assert reply == "Updated: Graduate AI Engineer at Example Ltd is now interview."
    listed = json.loads(tools["list_applications"].invoke({"status": "interview"}))
    assert [a["job_id"] for a in listed] == ["101"]


def test_invalid_status_is_rejected_by_the_tool_schema() -> None:
    tools, _ = _tools()
    # The Literal type becomes an enum in the schema, so the LLM is told the
    # allowed values up front, and a bad value is rejected before our code runs.
    schema = tools["list_applications"].args["status"]
    assert "interview" in json.dumps(schema)


def test_score_jobs_ranks_best_first_and_flags_unknown_ids() -> None:
    from jobagent.matching import Matcher
    from tests.fakes import FakeStructuredModel, match

    search = FakeSearch(jobs=[job("1", "Data Analyst"), job("2", "AI Engineer")])
    matcher = Matcher(FakeStructuredModel([match(40), match(85, ["Docker"])]), "cv")
    tools = {t.name: t for t in build_tools(search, Tracker(), matcher)}  # type: ignore[arg-type]
    tools["search_jobs"].invoke({"what": "ai"})

    out = json.loads(tools["score_jobs"].invoke({"job_ids": ["1", "2", "zzz"]}))

    assert [r["job_id"] for r in out] == ["2", "1", "zzz"]
    assert out[0]["score"] == 85 and out[0]["missing_skills"] == ["Docker"]
    assert "error" in out[2]


def test_score_jobs_is_only_offered_with_a_cv() -> None:
    assert "score_jobs" not in _tools()[0]
