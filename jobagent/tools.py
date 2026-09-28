"""The tools the agent can call.

Each tool is a plain function with a typed signature and a docstring: LangChain
turns those into the JSON schema the LLM sees, so the docstrings are written
for the model — they say *when* to use the tool, not just what it does.

Tools never raise into the agent loop. Failures come back as an ``Error: ...``
string, so the model can read what went wrong and recover (e.g. retry with
different keywords) instead of the whole run crashing.
"""

import json

from langchain_core.tools import BaseTool, tool

from jobagent.jobs import Job, JobSearchClient, JobSearchError
from jobagent.matching import Matcher
from jobagent.tracker import Status, Tracker


def _job_summary(job: Job) -> dict[str, object]:
    salary = None
    if job.salary_min or job.salary_max:
        salary = f"{job.salary_min or '?'}-{job.salary_max or '?'} EUR"
    return {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "salary": salary,
        "posted": job.created.date().isoformat() if job.created else None,
        "snippet": job.description[:300],
        "url": job.url,
    }


def build_tools(
    search: JobSearchClient, tracker: Tracker, matcher: Matcher | None = None
) -> list[BaseTool]:
    """Create the agent's tools, bound to a search client and a tracker.

    ``score_jobs`` is only offered when a ``matcher`` (i.e. a CV) is available,
    so the model is never shown a tool it can't use.

    Jobs returned by a search are remembered for the session, so the model can
    refer to them later by id alone (e.g. to save one) without re-sending them.
    """
    seen: dict[str, Job] = {}

    @tool
    def search_jobs(
        what: str, where: str = "", max_days_old: int | None = None, limit: int = 5
    ) -> str:
        """Search live job listings in Ireland.

        Use this whenever the user wants to find jobs. ``what`` is keywords such
        as "graduate ai engineer" or "junior data analyst python"; ``where`` is
        an optional place such as "Dublin" or "Cork"; ``max_days_old`` limits
        results to recent postings; ``limit`` is 1-20. Returns a JSON list of
        jobs with their ids.
        """
        try:
            jobs = search.search(what, where, max_days_old=max_days_old, limit=min(limit, 20))
        except (JobSearchError, ValueError) as exc:
            return f"Error: {exc}"
        for job in jobs:
            seen[job.id] = job
        if not jobs:
            return "No jobs found. Try broader or different keywords."
        return json.dumps([_job_summary(j) for j in jobs])

    @tool
    def save_application(job_id: str, notes: str = "") -> str:
        """Start tracking a job the user wants to apply to.

        Only use this when the user asks to save or track a job. ``job_id``
        must come from an earlier search_jobs result in this conversation.
        """
        job = seen.get(job_id)
        if job is None:
            return f"Error: unknown job id {job_id!r}. Search for it first."
        app = tracker.save(job, notes=notes)
        return f"Saved: {app.title} at {app.company} (status: {app.status})."

    @tool
    def update_application(job_id: str, status: Status, notes: str | None = None) -> str:
        """Change the status of a tracked application.

        ``status`` is one of: saved, applied, interview, offer, rejected. Use
        this when the user reports progress, e.g. "I applied to the Intel job".
        """
        try:
            app = tracker.update_status(job_id, status, notes)
        except (KeyError, ValueError) as exc:
            return f"Error: {exc}"
        return f"Updated: {app.title} at {app.company} is now {app.status}."

    @tool
    def list_applications(status: Status | None = None) -> str:
        """List the jobs the user is tracking, optionally only one status.

        Use this for questions like "what have I applied to?" or "show my
        interviews". Returns a JSON list, newest first.
        """
        apps = tracker.list(status)  # status already validated by the tool schema
        if not apps:
            return "No tracked applications yet."
        return json.dumps([a.model_dump() for a in apps])

    @tool
    def score_jobs(job_ids: list[str]) -> str:
        """Score how well jobs fit the user's CV, best match first.

        Use this when the user asks which jobs suit them, or to rank search
        results. ``job_ids`` must come from earlier search_jobs results. Returns
        a JSON list with a 0-100 score, matched and missing skills, and a reason.
        """
        assert matcher is not None  # tool only offered when a matcher exists
        results = []
        for job_id in job_ids:
            job = seen.get(job_id)
            if job is None:
                results.append({"job_id": job_id, "error": "unknown job id, search first"})
                continue
            match = matcher.score(job)
            results.append({"job_id": job_id, "title": job.title, **match.model_dump()})
        results.sort(key=lambda r: r.get("score", -1), reverse=True)
        return json.dumps(results)

    tools: list[BaseTool] = [search_jobs, save_application, update_application, list_applications]
    if matcher is not None:
        tools.append(score_jobs)
    return tools
