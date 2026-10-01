"""Score accuracy: does CV-match scoring land in a plausible range, and rank correctly?

Exact scores from an LLM judge aren't reproducible to the point, so the band
cases assert a range rather than a number; the rank pairs check relative
ordering, which holds even when the exact band is a judgement call.
"""

from dataclasses import dataclass

from evals.cases import RANK_PAIRS, SCORE_CASES
from jobagent.agent import build_llm
from jobagent.config import Settings, get_settings
from jobagent.cv import load_cv
from jobagent.matching import Matcher


@dataclass
class BandResult:
    """One scored job vs. the band it was expected to land in."""

    name: str
    expected_min: int
    expected_max: int
    got: int

    @property
    def correct(self) -> bool:
        return self.expected_min <= self.got <= self.expected_max


@dataclass
class RankResult:
    """One pair of jobs and whether the expected one scored higher."""

    name: str
    higher_score: int
    lower_score: int

    @property
    def correct(self) -> bool:
        return self.higher_score > self.lower_score


def _matcher(settings: Settings | None = None) -> Matcher:
    s = settings or get_settings()
    if not s.cv_path:
        raise RuntimeError("CV_PATH must be set in .env to run the score eval")
    return Matcher(build_llm(s), load_cv(s.cv_path))  # type: ignore[arg-type]


def run_bands(settings: Settings | None = None) -> list[BandResult]:
    """Score each labelled job and check it lands in its expected band."""
    matcher = _matcher(settings)
    return [
        BandResult(case.name, case.expected_min, case.expected_max, matcher.score(case.job).score)
        for case in SCORE_CASES
    ]


def run_ranks(settings: Settings | None = None) -> list[RankResult]:
    """Score each pair and check the better-fit job scores higher."""
    matcher = _matcher(settings)
    results = []
    for name, better, worse in RANK_PAIRS:
        results.append(RankResult(name, matcher.score(better).score, matcher.score(worse).score))
    return results
