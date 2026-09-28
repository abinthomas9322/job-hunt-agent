"""Chat with the agent in the terminal.

Run from the repo root (needs GROQ_API_KEY and the Adzuna keys in .env)::

    python -m jobagent.cli
    python -m jobagent.cli --trace     # also print every tool call and result
"""

import argparse
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from jobagent.agent import ask, build_agent, build_llm, final_text
from jobagent.config import get_settings
from jobagent.jobs import JobSearchClient
from jobagent.tools import build_tools
from jobagent.tracker import Tracker


def main() -> None:  # pragma: no cover - interactive shell around tested pieces
    parser = argparse.ArgumentParser(description="Chat with the Job Hunt Agent.")
    parser.add_argument("--trace", action="store_true", help="show tool calls and results")
    args = parser.parse_args()

    settings = get_settings()
    tools = build_tools(JobSearchClient(settings), Tracker(settings.db_path))
    agent = build_agent(build_llm(settings), tools)
    thread = uuid.uuid4().hex
    print("Job Hunt Agent — type 'quit' to exit.\n")

    while (text := input("you> ").strip()).lower() not in {"quit", "exit"}:
        if not text:
            continue
        messages = ask(agent, text, thread, max_steps=settings.max_steps)
        if args.trace:
            for msg in messages:
                if isinstance(msg, AIMessage):
                    for call in msg.tool_calls:
                        print(f"  → {call['name']}({call['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  ← {str(msg.content)[:200]}")
        print(f"\nagent> {final_text(messages)}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
