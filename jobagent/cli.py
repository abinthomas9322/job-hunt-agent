"""Chat with the agent in the terminal.

Run from the repo root (needs GROQ_API_KEY and the Adzuna keys in .env)::

    python -m jobagent.cli
    python -m jobagent.cli --trace     # also print every tool call and result
"""

import argparse
import sys
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from jobagent.agent import ask, build_agent, build_llm, final_text, pending_approval, resume
from jobagent.config import get_settings
from jobagent.cv import load_cv
from jobagent.letters import LetterWriter
from jobagent.matching import Matcher
from jobagent.sources import build_job_source
from jobagent.tools import build_tools
from jobagent.tracker import Tracker


def main() -> None:  # pragma: no cover - interactive shell around tested pieces
    parser = argparse.ArgumentParser(description="Chat with the Job Hunt Agent.")
    parser.add_argument("--trace", action="store_true", help="show tool calls and results")
    args = parser.parse_args()
    # LLM replies can contain characters the Windows console codepage lacks.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    settings = get_settings()
    llm = build_llm(settings)
    cv = load_cv(settings.cv_path) if settings.cv_path else None
    matcher = Matcher(llm, cv) if cv else None  # type: ignore[arg-type]
    writer = LetterWriter(llm, cv) if cv else None  # type: ignore[arg-type]
    tools = build_tools(
        build_job_source(settings), Tracker(settings.db_path), matcher, writer, settings.letters_dir
    )
    agent = build_agent(llm, tools)
    thread = uuid.uuid4().hex
    cv_note = "CV loaded" if matcher else "no CV_PATH set, so job scoring is off"
    print(f"Job Hunt Agent ({cv_note}). Type 'quit' to exit.\n")

    while (text := input("you> ").strip()).lower() not in {"quit", "exit"}:
        if not text:
            continue
        messages = ask(agent, text, thread, max_steps=settings.max_steps)
        # A tool paused for approval (a cover-letter draft): ask, then resume.
        while (pending := pending_approval(agent, thread)) is not None:
            print(f"\n--- Draft cover letter: {pending['title']} at {pending['company']} ---")
            print(pending["draft"])
            choice = input("\nApprove? [y]es / [n]o, with feedback / [e]dit: ").strip().lower()
            if choice.startswith("y"):
                decision: dict[str, object] = {"approved": True}
            elif choice.startswith("e"):
                print("Paste your edited letter, then a line with just END:")
                lines = iter(input, "END")
                decision = {"approved": True, "text": "\n".join(lines)}
            else:
                decision = {"approved": False, "feedback": input("What should change? ")}
            messages += resume(agent, decision, thread, max_steps=settings.max_steps)
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
