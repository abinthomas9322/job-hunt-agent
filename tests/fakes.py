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


class FakeStructuredModel:
    """Fake chat model for structured output: returns canned results in order."""

    def __init__(self, results: Sequence[dict[str, Any] | Exception]) -> None:
        self.results = list(results)
        self.schema: type | None = None
        self.prompts: list[Any] = []

    def with_structured_output(self, schema: type, **kwargs: Any) -> "FakeStructuredModel":
        self.schema = schema
        return self

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(messages)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def match(score: int, missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "score": score,
        "matched_skills": ["Python", "RAG"],
        "missing_skills": missing or [],
        "summary": f"Scored {score}.",
    }


class FakeChat:
    """Fake plain chat model: returns canned replies (or raises) in order."""

    def __init__(self, replies: Sequence[str | Exception]) -> None:
        self.replies = list(replies)
        self.prompts: list[Any] = []

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(reply)
