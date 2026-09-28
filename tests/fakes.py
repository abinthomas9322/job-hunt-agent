"""Test doubles for the LLM and the job search API."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from jobagent.jobs import Job, JobSearchError


class ScriptedModel:
    """A fake tool-calling chat model that replies from a fixed script.

    Each ``invoke`` returns the next scripted ``AIMessage`` and records the
    messages it was given, so tests can check what the agent sent the LLM.
    """

    def __init__(self, replies: Sequence[AIMessage]) -> None:
        self.replies = list(replies)
        self.calls: list[list[BaseMessage]] = []
        self.bound: list[str] = []

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "ScriptedModel":
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        return self.replies.pop(0)


def call(name: str, call_id: str = "c1", **args: Any) -> AIMessage:
    """An AI message asking to run one tool."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def job(job_id: str = "101", title: str = "Graduate AI Engineer") -> Job:
    return Job(
        id=job_id,
        title=title,
        company="Example Ltd",
        location="Dublin",
        description="Python, LLMs and RAG.",
        url=f"https://example.test/{job_id}",
        created="2026-09-20T10:00:00Z",  # type: ignore[arg-type]
        salary_min=40000,
    )


class FakeSearch:
    """Stands in for JobSearchClient; returns canned jobs or a canned error."""

    def __init__(self, jobs: list[Job] | None = None, error: str | None = None) -> None:
        self.jobs = jobs if jobs is not None else [job()]
        self.error = error
        self.queries: list[dict[str, Any]] = []

    def search(self, what: str, where: str = "", **kwargs: Any) -> list[Job]:
        self.queries.append({"what": what, "where": where, **kwargs})
        if self.error:
            raise JobSearchError(self.error)
        return self.jobs
