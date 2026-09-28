"""Tests for the agent loop, driven by a scripted fake LLM."""

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from jobagent.agent import SYSTEM_PROMPT, ask, build_agent, build_llm, final_text
from jobagent.config import Settings
from jobagent.tools import build_tools
from jobagent.tracker import Tracker
from tests.fakes import FakeSearch, ScriptedModel, call


def _agent(model: ScriptedModel, search: FakeSearch | None = None):  # type: ignore[no-untyped-def]
    tools = build_tools(search or FakeSearch(), Tracker())
    return build_agent(model, tools)


def test_agent_calls_a_tool_then_answers() -> None:
    model = ScriptedModel(
        [
            call("search_jobs", what="graduate ai engineer", where="Dublin"),
            AIMessage("Found 1 job: Graduate AI Engineer at Example Ltd."),
        ]
    )
    search = FakeSearch()
    messages = ask(_agent(model, search), "Find AI grad jobs in Dublin", "t1")

    assert search.queries[0]["what"] == "graduate ai engineer"
    assert [type(m).__name__ for m in messages] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert isinstance(messages[2], ToolMessage) and "Example Ltd" in str(messages[2].content)
    assert final_text(messages) == "Found 1 job: Graduate AI Engineer at Example Ltd."
    # The LLM saw the system prompt first, then the tool result on its 2nd turn.
    assert isinstance(model.calls[0][0], SystemMessage)
    assert model.calls[0][0].content == SYSTEM_PROMPT
    assert isinstance(model.calls[1][-1], ToolMessage)
    assert model.bound == [
        "search_jobs",
        "save_application",
        "update_application",
        "list_applications",
    ]


def test_agent_remembers_earlier_turns_on_the_same_thread() -> None:
    model = ScriptedModel(
        [
            call("search_jobs", what="data analyst"),
            AIMessage("Here is one job."),
            call("save_application", call_id="c2", job_id="101"),
            AIMessage("Saved it."),
        ]
    )
    agent = _agent(model)
    ask(agent, "Find data analyst jobs", "same")
    second = ask(agent, "Save the first one", "same")

    # Only the new turn is returned, but the LLM received the whole history.
    assert [type(m).__name__ for m in second][0] == "HumanMessage"
    assert "Saved: Graduate AI Engineer" in str(second[2].content)
    assert len(model.calls[-1]) > len(second)


def test_threads_are_isolated() -> None:
    model = ScriptedModel([AIMessage("hi A"), AIMessage("hi B")])
    agent = _agent(model)
    ask(agent, "hello", "A")
    ask(agent, "hello", "B")
    assert len(model.calls[1]) == 2  # system + this thread's single message


def test_step_limit_stops_a_looping_agent() -> None:
    looping = ScriptedModel([call("list_applications", call_id=f"c{i}") for i in range(20)])
    with pytest.raises(Exception, match="[Rr]ecursion"):
        ask(_agent(looping), "loop", "t", max_steps=4)


def test_final_text_is_empty_without_a_plain_ai_reply() -> None:
    assert final_text([call("search_jobs", what="x")]) == ""


def test_build_llm_requires_key_and_uses_groq_settings() -> None:
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_llm(Settings(_env_file=None, groq_api_key=""))  # type: ignore[call-arg]
    llm = build_llm(Settings(_env_file=None, groq_api_key="k", llm_model="m"))  # type: ignore[call-arg]
    assert llm.model_name == "m"  # type: ignore[attr-defined]
    assert "groq.com" in str(llm.openai_api_base)  # type: ignore[attr-defined]
