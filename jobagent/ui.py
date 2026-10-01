"""A chat UI for the agent, in the browser.

Run from the repo root (needs GROQ_API_KEY and a job source's keys in .env)::

    streamlit run jobagent/ui.py

Mirrors jobagent.cli's flow (ask -> pending_approval -> resume), but as a
Streamlit app: the LangGraph checkpointer already holds each thread's full
message history, so this module only keeps a small display-only transcript
to redraw on every rerun (Streamlit reruns the whole script per interaction).
"""

import uuid
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from jobagent.agent import ask, build_agent, build_llm, final_text, pending_approval, resume
from jobagent.config import get_settings
from jobagent.cv import load_cv
from jobagent.letters import LetterWriter
from jobagent.matching import Matcher
from jobagent.sources import build_job_source
from jobagent.tools import build_tools
from jobagent.tracker import Tracker

st.set_page_config(page_title="Job Hunt Agent", page_icon="\U0001f9ed")


@st.cache_resource
def _build() -> tuple[CompiledStateGraph[Any], bool]:
    """Build the agent once per server process; returns (agent, cv_loaded)."""
    settings = get_settings()
    llm = build_llm(settings)
    cv = load_cv(settings.cv_path) if settings.cv_path else None
    matcher = Matcher(llm, cv) if cv else None  # type: ignore[arg-type]
    writer = LetterWriter(llm, cv) if cv else None  # type: ignore[arg-type]
    tools = build_tools(
        build_job_source(settings),
        Tracker(settings.db_path),
        matcher,
        writer,
        settings.letters_dir,
    )
    return build_agent(llm, tools), cv is not None


def _render_trace(messages: list[BaseMessage]) -> None:
    """Show tool calls and their results for one turn, collapsed by default."""
    calls = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
    if not calls:
        return
    with st.expander(f"{sum(len(m.tool_calls) for m in calls)} tool call(s)"):
        for call_msg in calls:
            for call in call_msg.tool_calls:
                st.code(f"{call['name']}({call['args']})", language="text")
        for result in messages:
            if isinstance(result, ToolMessage):
                st.code(str(result.content)[:500], language="text")


def _init_state() -> None:
    if "thread" not in st.session_state:
        st.session_state.thread = uuid.uuid4().hex
        st.session_state.history = []


def _add_turn(role: str, text: str, trace: list[BaseMessage] | None = None) -> None:
    st.session_state.history.append({"role": role, "text": text, "trace": trace})


def main() -> None:  # pragma: no cover - interactive app around tested pieces
    agent, has_cv = _build()
    settings = get_settings()
    _init_state()

    st.title("\U0001f9ed Job Hunt Agent")
    st.caption("CV loaded — job scoring is on" if has_cv else "No CV_PATH set — job scoring is off")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.write(turn["text"])
            if turn["trace"]:
                _render_trace(turn["trace"])

    pending = pending_approval(agent, st.session_state.thread)
    if pending is not None:
        with st.chat_message("assistant"):
            st.write(f"Draft cover letter for **{pending['title']}** at **{pending['company']}**:")
            draft = st.text_area(
                "Draft", value=pending["draft"], height=260, label_visibility="collapsed"
            )
            feedback = st.text_input("Feedback if you reject it (optional)")
            col1, col2 = st.columns(2)
            approve = col1.button("Approve", type="primary", use_container_width=True)
            reject = col2.button("Reject", use_container_width=True)
        decision: dict[str, Any] | None = None
        if approve:
            decision = {"approved": True, "text": draft}
        elif reject:
            decision = {"approved": False, "feedback": feedback or "none given"}
        if decision is not None:
            messages = resume(
                agent, decision, st.session_state.thread, max_steps=settings.max_steps
            )
            _add_turn("assistant", final_text(messages), messages)
            st.rerun()
        return  # hold the chat input while a decision is pending

    if text := st.chat_input("Ask about jobs, applications or cover letters..."):
        _add_turn("user", text)
        with st.spinner("Thinking..."):
            messages = ask(agent, text, st.session_state.thread, max_steps=settings.max_steps)
        _add_turn("assistant", final_text(messages), messages)
        st.rerun()


main()
