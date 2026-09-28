"""Tests for CV-to-job match scoring (LLM replaced by a fake)."""

import pytest
from pydantic import ValidationError

from jobagent.matching import RUBRIC, Matcher, MatchResult
from tests.fakes import FakeStructuredModel, job, match


def test_score_returns_validated_result_and_sends_cv_and_job() -> None:
    llm = FakeStructuredModel([match(72, ["Docker"])])
    result = Matcher(llm, "CV: Python, RAG").score(job())

    assert result == MatchResult(**match(72, ["Docker"]))
    assert llm.schema is MatchResult
    system, user = llm.prompts[0]
    assert system == ("system", RUBRIC)
    assert "CV: Python, RAG" in user[1] and "Graduate AI Engineer" in user[1]


def test_out_of_range_score_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Matcher(FakeStructuredModel([match(140)]), "cv").score(job())


def test_empty_cv_is_rejected() -> None:
    with pytest.raises(ValueError):
        Matcher(FakeStructuredModel([]), "  ")
