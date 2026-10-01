"""The agent: an LLM in a loop that decides which tool to call next.

The graph has two nodes::

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                   \\--(no tool calls: final answer)------------> END

``agent`` asks the LLM what to do given the conversation so far. If it replies
with tool calls, ``tools`` runs them and appends the results, and control goes
back to ``agent``; if it replies with plain text, that is the answer. A
checkpointer stores each conversation's messages by ``thread_id``, which is
what gives the agent memory across turns.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from jobagent.config import Settings, get_settings

SYSTEM_PROMPT = """You are a job-search assistant for a graduate in Ireland.

You help the user find real job listings, keep track of applications and plan
next steps. Rules:
- Use the tools for facts. Never invent jobs, companies, salaries or links:
  only mention jobs that a search_jobs result actually returned.
- When you list jobs, include the title, company, location and the url.
- Only mention fit scores that score_jobs returned.
- Only save or update applications when the user asks you to.
- If a tool returns an error, explain it briefly and try a sensible fix
  (for example broader keywords) at most once.
- Be concise."""


# Added only when a CV is loaded, i.e. when the score_jobs tool exists.
CV_PROMPT = """
The user's CV is already loaded and score_jobs compares jobs against it. Never
ask the user for their CV or skills: when they ask which jobs fit them, search
first, then call score_jobs with the job ids and report each job's score, main
missing skills and url.
Only draft a cover letter (draft_cover_letter) when the user asks for one. The
user approves each draft, so never say a letter is saved until the tool says so."""


def system_prompt(tools: Sequence[BaseTool]) -> str:
    """The system prompt, with CV instructions only if score_jobs is available."""
    has_cv = any(t.name == "score_jobs" for t in tools)
    return SYSTEM_PROMPT + (CV_PROMPT if has_cv else "")


class ToolCallingModel(Protocol):
    """The part of a LangChain chat model the agent needs."""

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> Any: ...


def build_llm(settings: Settings | None = None) -> ToolCallingModel:
    """Create the Groq chat model through its OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    s = settings or get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY must be set in .env")
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.groq_api_key,  # type: ignore[arg-type]
        temperature=0,
        # Groq's free tier allows ~8k tokens/minute; back off and retry on 429s
        # instead of failing the whole run.
        max_retries=6,
    )


def build_agent(
    llm: ToolCallingModel,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any]:
    """Wire the LLM and tools into a LangGraph tool-calling loop."""
    model = llm.bind_tools(tools)
    prompt = system_prompt(tools)

    def agent_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(prompt), *state["messages"]]
        return {"messages": [model.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def ask(
    agent: CompiledStateGraph[Any], message: str, thread_id: str, max_steps: int = 12
) -> list[BaseMessage]:
    """Send one user message on a conversation thread; return the new messages.

    The returned list starts with the user's message and ends with the agent's
    final reply, with any tool calls and tool results in between — which is
    what the CLI shows and what the evals inspect.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_steps,
    }
    before = len(agent.get_state(config).values.get("messages", []))
    result = agent.invoke({"messages": [("user", message)]}, config)
    return list(result["messages"][before:])


def pending_approval(agent: CompiledStateGraph[Any], thread_id: str) -> dict[str, Any] | None:
    """What the paused agent is waiting for the user to approve, if anything.

    A tool such as ``draft_cover_letter`` pauses the graph with an interrupt;
    its payload (e.g. the draft letter) is returned here until the thread is
    resumed with ``resume``.
    """
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    return dict(state.interrupts[0].value) if state.interrupts else None


def resume(
    agent: CompiledStateGraph[Any], decision: dict[str, Any], thread_id: str, max_steps: int = 12
) -> list[BaseMessage]:
    """Answer a pending approval and let the agent carry on; return the new messages.

    ``decision`` is ``{"approved": True}`` (optionally with an edited ``text``)
    or ``{"approved": False, "feedback": "..."}``.
    """
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_steps,
    }
    before = len(agent.get_state(config).values.get("messages", []))
    result = agent.invoke(Command(resume=decision), config)
    return list(result["messages"][before:])


def final_text(messages: Sequence[BaseMessage]) -> str:
    """The text of the last AI message (the agent's answer to the user)."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return str(msg.content)
    return ""
