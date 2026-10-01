"""Draft cover letters from the CV, and save the ones the user approves.

Drafting is a plain LLM call with strict rules: only facts that are in the CV,
a fixed length and structure. Nothing is saved until the user approves the
draft; that approval step lives in the agent's ``draft_cover_letter`` tool,
which pauses the graph (a LangGraph *interrupt*) and waits for the human.
"""

import re
from pathlib import Path
from typing import Any, Protocol

from jobagent.jobs import Job
from jobagent.matching import MAX_JOB_CHARS

LETTER_RULES = """You write cover letters for a job seeker in Ireland.

Rules:
- Use ONLY facts from the CANDIDATE CV. Never invent employers, projects,
  numbers, grades or skills. If the job asks for something the CV lacks, do not
  claim it; show related evidence or willingness to learn instead.
- 200-280 words, Irish/UK English, plain and confident, no clichés such as
  "I am writing to express my interest".
- Structure: why this role at this company (one short paragraph), two or three
  concrete pieces of evidence from the CV matched to the job's requirements,
  a short closing with availability.
- Start with "Dear Hiring Team," and end with "Kind regards," and the
  candidate's name if the CV gives it.
- Return only the letter text."""


class ChatModel(Protocol):
    """The part of a LangChain chat model the writer needs."""

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any: ...


class LetterWriter:
    """Drafts cover letters for one CV."""

    def __init__(self, llm: ChatModel, cv_text: str) -> None:
        if not cv_text.strip():
            raise ValueError("cv_text must not be empty")
        self.cv_text = cv_text
        self._llm = llm

    def draft(self, job: Job, focus: str = "") -> str:
        """Write a draft letter for ``job``; ``focus`` is extra guidance from the user."""
        job_text = (
            f"Title: {job.title}\nCompany: {job.company}\nLocation: {job.location}\n"
            f"Description: {job.description[:MAX_JOB_CHARS]}"
        )
        extra = f"\n\nUSER'S GUIDANCE FOR THIS DRAFT:\n{focus}" if focus.strip() else ""
        reply = self._llm.invoke(
            [
                ("system", LETTER_RULES),
                ("user", f"CANDIDATE CV:\n{self.cv_text}\n\nJOB:\n{job_text}{extra}"),
            ]
        )
        text = str(getattr(reply, "content", reply)).strip()
        if not text:
            raise ValueError("the model returned an empty letter")
        return text


def save_letter(letters_dir: str, job: Job, text: str) -> Path:
    """Write an approved letter to ``<letters_dir>/<company>-<job id>.md``."""
    slug = re.sub(r"[^a-z0-9]+", "-", f"{job.company}-{job.id}".lower()).strip("-")
    path = Path(letters_dir) / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {job.title} at {job.company}\n\n{job.url}\n\n{text}\n", encoding="utf-8")
    return path
