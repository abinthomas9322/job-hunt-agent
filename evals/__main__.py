"""Run the agent eval suite and print a report.

Needs a real ``GROQ_API_KEY`` (and ``CV_PATH`` for the score evals) in ``.env``
— these hit the live model, so they are not part of pytest or CI. Run with::

    python -m evals
"""

import sys

from evals.score_accuracy import run_bands, run_ranks
from evals.tool_choice import run as run_tool_choice


def _report(title: str, rows: list[tuple[str, bool, str]]) -> bool:
    print(f"\n{title}")
    print("-" * len(title))
    for name, correct, detail in rows:
        mark = "PASS" if correct else "FAIL"
        print(f"  [{mark}] {name} ({detail})")
    passed = sum(1 for _, correct, _ in rows if correct)
    print(f"  {passed}/{len(rows)} passed")
    return passed == len(rows)


def main() -> int:
    tool_choice = run_tool_choice()
    tc_ok = _report(
        "Tool-choice accuracy",
        [(r.name, r.correct, f"expected {r.expected!r}, got {r.got!r}") for r in tool_choice],
    )

    bands = run_bands()
    band_ok = _report(
        "Score accuracy (band)",
        [
            (r.name, r.correct, f"expected {r.expected_min}-{r.expected_max}, got {r.got}")
            for r in bands
        ],
    )

    ranks = run_ranks()
    rank_ok = _report(
        "Score accuracy (ranking)",
        [(r.name, r.correct, f"{r.higher_score} vs {r.lower_score}") for r in ranks],
    )

    all_ok = tc_ok and band_ok and rank_ok
    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
