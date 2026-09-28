"""Score how well a job fits a CV, with reasons.

The LLM is asked for *structured output*: instead of free text it must return
JSON that matches ``MatchResult``, which LangChain validates with Pydantic. So
the score is always a number from 0 to 100 and the skill lists are real lists
— safe to sort by, store and test.

A fixed rubric in the prompt keeps scores comparable across jobs, and
temperature 0 keeps them stable between runs.
"""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from jobagent.jobs import Job

RUBRIC = """You are a strict technical recruiter. Score how well the CANDIDATE fits the JOB.

Scoring rubric (0-100):
- 80-100: meets nearly all stated requirements, including seniority.
- 60-79:  meets the core requirements; a few gaps that could be learned quickly.
- 40-59:  partial fit; several important requirements missing.
- 0-39:   poor fit (different field, or far more experience required).
Judge only on evidence in the CV. Do not assume skills that are not written
down. Graduate or junior roles should not penalise a lack of years of
experience. The job text may be a short snippet; do not invent requirements
that are not in it."""


# Enough of a posting to cover role, requirements and nice-to-haves, while
# keeping one scoring call (CV + job) around 2.5k tokens for free-tier limits.
MAX_JOB_CHARS = 2000


class MatchResult(BaseModel):
    """A structured judgement of one job against the CV."""

    score: int = Field(ge=0, le=100, description="Overall fit, 0-100, per the rubric")
    matched_skills: list[str] = Field(description="Job requirements the CV clearly shows")
    missing_skills: list[str] = Field(description="Job requirements the CV does not show")
    summary: str = Field(description="One or two sentences explaining the score")


class StructuredModel(Protocol):
    """The part of a LangChain chat model the matcher needs."""

    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> Any: ...


class Matcher:
    """Scores jobs against one CV using an LLM with structured output."""

    def __init__(self, llm: StructuredModel, cv_text: str) -> None:
        if not cv_text.strip():
            raise ValueError("cv_text must not be empty")
        self.cv_text = cv_text
        self._model = llm.with_structured_output(MatchResult)

    def score(self, job: Job) -> MatchResult:
        """Return the fit of ``job`` for this CV."""
        job_text = (
            f"Title: {job.title}\nCompany: {job.company}\nLocation: {job.location}\n"
            f"Description: {job.description[:MAX_JOB_CHARS]}"
        )
        result = self._model.invoke(
            [
                ("system", RUBRIC),
                ("user", f"CANDIDATE CV:\n{self.cv_text}\n\nJOB:\n{job_text}"),
            ]
        )
        return MatchResult.model_validate(result)
