"""Hand-labelled cases for the eval harness: what each prompt should trigger."""

import json
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from jobagent.jobs import Job
from jobagent.tools import _job_summary

# A few fixed jobs, used both to seed prior "search" turns and to score.
AI_JOB = Job(
    id="ai-1",
    title="Graduate AI Engineer",
    company="Datadog",
    location="Dublin",
    description=(
        "Build and ship LLM-powered features. Requirements: Python, experience with "
        "retrieval-augmented generation (RAG), vector search, and shipping production ML "
        "systems. New grads welcome."
    ),
    url="https://example.test/ai-1",
)

RETAIL_JOB = Job(
    id="retail-1",
    title="Retail Supervisor",
    company="Centra",
    location="Cork",
    description=(
        "Lead a small store team, manage stock and tills, and deliver great customer "
        "service in a fast-paced grocery store."
    ),
    url="https://example.test/retail-1",
)

PILOT_JOB = Job(
    id="pilot-1",
    title="Commercial Airline First Officer",
    company="Aer Lingus",
    location="Dublin",
    description=(
        "Requires a current ATPL licence, 1500+ logged flight hours and a Class 1 medical "
        "certificate. Type rating on Airbus A320 family preferred."
    ),
    url="https://example.test/pilot-1",
)

JAVA_JOB = Job(
    id="java-1",
    title="Senior Java Backend Engineer",
    company="MongoDB",
    location="Dublin",
    description=(
        "8+ years building distributed Java systems at scale. Deep expertise in JVM "
        "internals, Kubernetes and Kafka required."
    ),
    url="https://example.test/java-1",
)


def _search_turn(job: Job, call_id: str = "c1") -> list[BaseMessage]:
    """A fake prior turn: the agent searched and got one job back, by ``job.id``."""
    return [
        AIMessage(
            "", tool_calls=[{"name": "search_jobs", "args": {"what": job.title}, "id": call_id}]
        ),
        ToolMessage(json.dumps([_job_summary(job)]), tool_call_id=call_id),
    ]


@dataclass
class ToolChoiceCase:
    """One labelled example: a conversation and the tool it should trigger (or none)."""

    name: str
    messages: list[BaseMessage]
    expected: str | None  # tool name, or None if the agent should just answer


TOOL_CHOICE_CASES: list[ToolChoiceCase] = [
    ToolChoiceCase(
        "search: plain request",
        [HumanMessage("Find graduate AI engineer jobs in Dublin")],
        "search_jobs",
    ),
    ToolChoiceCase(
        "save: after a search",
        [*_search_turn(AI_JOB), HumanMessage("Save that one, I want to apply")],
        "save_application",
    ),
    ToolChoiceCase(
        "update: reporting progress on a tracked job",
        [
            *_search_turn(AI_JOB),
            HumanMessage("Save that one, I want to apply"),
            AIMessage("Saved: Graduate AI Engineer at Datadog (status: saved)."),
            HumanMessage("I had an interview for the Datadog job today"),
        ],
        "update_application",
    ),
    ToolChoiceCase(
        "list: asking what's tracked",
        [HumanMessage("What have I applied to so far?")],
        "list_applications",
    ),
    ToolChoiceCase(
        "score: asking whether a job fits",
        [*_search_turn(AI_JOB), HumanMessage("Does this job suit my background?")],
        "score_jobs",
    ),
    ToolChoiceCase(
        "cover letter: explicit request",
        [*_search_turn(AI_JOB), HumanMessage("Write me a cover letter for that role")],
        "draft_cover_letter",
    ),
    ToolChoiceCase(
        "no tool: greeting",
        [HumanMessage("Hi, what can you help me with?")],
        None,
    ),
    ToolChoiceCase(
        "no tool: general knowledge, not a job-search task",
        [HumanMessage("In one sentence, what does RAG stand for?")],
        None,
    ),
]


@dataclass
class ScoreCase:
    """A job with the score band the real CV should land in, per the rubric."""

    name: str
    job: Job
    expected_min: int
    expected_max: int


SCORE_CASES: list[ScoreCase] = [
    ScoreCase("strong fit: AI/RAG grad role", AI_JOB, 55, 100),
    ScoreCase("strong fit: retail supervisor (real past experience)", RETAIL_JOB, 50, 100),
    ScoreCase("poor fit: commercial pilot", PILOT_JOB, 0, 25),
    ScoreCase("poor fit: senior Java/distributed systems", JAVA_JOB, 0, 35),
]

# Pairs where the first should out-score the second — checks relative ranking,
# which is more robust than an exact band when the rubric is a judgement call.
RANK_PAIRS: list[tuple[str, Job, Job]] = [
    ("AI role beats an unrelated pilot role", AI_JOB, PILOT_JOB),
    ("AI role beats a senior-Java role needing years the CV doesn't show", AI_JOB, JAVA_JOB),
]


def eval_tools() -> list[BaseTool]:
    """Build the full 6-tool set (CV tools included) for binding to the real LLM.

    Tool-choice cases only inspect which tool the model *asks* to call — nothing
    here is actually executed, so the search/tracker/matcher/writer backends are
    throwaway fakes that just need the right shape for ``build_tools`` to accept.
    """
    from jobagent.letters import LetterWriter
    from jobagent.matching import Matcher
    from jobagent.tools import build_tools
    from jobagent.tracker import Tracker
    from tests.fakes import FakeChat, FakeSearch, FakeStructuredModel

    matcher = Matcher(FakeStructuredModel([]), "placeholder cv text")
    writer = LetterWriter(FakeChat([]), "placeholder cv text")
    return build_tools(FakeSearch([AI_JOB]), Tracker(), matcher, writer)
