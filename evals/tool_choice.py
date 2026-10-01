"""Tool-choice accuracy: does the real LLM pick the right tool for each prompt?"""

from dataclasses import dataclass

from langchain_core.messages import SystemMessage

from evals.cases import TOOL_CHOICE_CASES, eval_tools
from jobagent.agent import build_llm, system_prompt
from jobagent.config import Settings


@dataclass
class ToolChoiceResult:
    """What one case expected vs. what the real model actually called."""

    name: str
    expected: str | None
    got: str | None

    @property
    def correct(self) -> bool:
        return self.got == self.expected


def run(settings: Settings | None = None) -> list[ToolChoiceResult]:
    """Run every tool-choice case once against the real LLM."""
    tools = eval_tools()
    model = build_llm(settings).bind_tools(tools)
    prompt = system_prompt(tools)
    results = []
    for case in TOOL_CHOICE_CASES:
        reply = model.invoke([SystemMessage(prompt), *case.messages])
        got = reply.tool_calls[0]["name"] if reply.tool_calls else None
        results.append(ToolChoiceResult(case.name, case.expected, got))
    return results
