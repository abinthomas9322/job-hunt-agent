"""Tests for cover-letter drafting, saving and the human-approval loop."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from jobagent.agent import ask, build_agent, final_text, pending_approval, resume
from jobagent.letters import LETTER_RULES, LetterWriter, save_letter
from jobagent.mcp_server import build_server
from jobagent.tools import build_tools
from jobagent.tracker import Tracker
from tests.fakes import FakeChat, FakeSearch, ScriptedModel, call, job

CV = "Abin. MSc AI. Built a RAG tutor with FastAPI."


def test_writer_sends_rules_cv_job_and_focus() -> None:
    llm = FakeChat(["Dear Hiring Team, ..."])
    letter = LetterWriter(llm, CV).draft(job(), focus="stress RAG")
    assert letter == "Dear Hiring Team, ..."
    system, user = llm.prompts[0]
    assert system == ("system", LETTER_RULES)
    assert CV in user[1] and "Graduate AI Engineer" in user[1] and "stress RAG" in user[1]


def test_writer_omits_guidance_when_no_focus_and_rejects_bad_input() -> None:
    llm = FakeChat(["Letter", "   "])
    writer = LetterWriter(llm, CV)
    writer.draft(job())
    assert "GUIDANCE" not in llm.prompts[0][1][1]
    with pytest.raises(ValueError, match="empty letter"):
        writer.draft(job())
    with pytest.raises(ValueError, match="cv_text"):
        LetterWriter(llm, "  ")


def test_save_letter_writes_markdown_with_a_safe_name(tmp_path: Path) -> None:
    path = save_letter(str(tmp_path), job(), "Dear Hiring Team")
    assert path.name == "example-ltd-101.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Graduate AI Engineer at Example Ltd")
    assert "https://example.test/101" in text and "Dear Hiring Team" in text


def _agent(model: ScriptedModel, drafts: list[str | Exception], letters: Path):  # type: ignore[no-untyped-def]
    writer = LetterWriter(FakeChat(drafts), CV)
    tools = build_tools(FakeSearch(), Tracker(), writer=writer, letters_dir=str(letters))
    return build_agent(model, tools)


def _search_then_draft(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(
        [call("search_jobs", what="ai"), call("draft_cover_letter", "c2", job_id="101"), *replies]
    )


def test_agent_pauses_for_approval_then_saves_the_approved_draft(tmp_path: Path) -> None:
    agent = _agent(_search_then_draft(AIMessage("Saved your letter.")), ["Draft 1"], tmp_path)
    ask(agent, "Write a cover letter for the AI job", "t")

    pending = pending_approval(agent, "t")
    assert pending == {
        "job_id": "101",
        "title": "Graduate AI Engineer",
        "company": "Example Ltd",
        "draft": "Draft 1",
    }
    assert not list(tmp_path.iterdir())  # nothing saved before approval

    messages = resume(agent, {"approved": True}, "t")
    assert pending_approval(agent, "t") is None
    assert isinstance(messages[0], ToolMessage)
    assert "Approved and saved to" in str(messages[0].content)
    assert "Draft 1" in (tmp_path / "example-ltd-101.md").read_text(encoding="utf-8")
    assert final_text(messages) == "Saved your letter."


def test_user_edits_are_what_gets_saved(tmp_path: Path) -> None:
    agent = _agent(_search_then_draft(AIMessage("Done.")), ["Draft 1"], tmp_path)
    ask(agent, "cover letter please", "t")
    resume(agent, {"approved": True, "text": "My own version"}, "t")
    saved = (tmp_path / "example-ltd-101.md").read_text(encoding="utf-8")
    assert "My own version" in saved and "Draft 1" not in saved


def test_rejection_feedback_goes_back_to_the_agent(tmp_path: Path) -> None:
    model = _search_then_draft(
        call("draft_cover_letter", "c3", job_id="101", focus="shorter"), AIMessage("Saved v2.")
    )
    agent = _agent(model, ["Draft 1", "Draft 2"], tmp_path)
    ask(agent, "cover letter please", "t")

    messages = resume(agent, {"approved": False, "feedback": "shorter"}, "t")
    assert str(messages[0].content) == "The user rejected the draft. Feedback: shorter"
    assert pending_approval(agent, "t")["draft"] == "Draft 2"  # type: ignore[index]

    resume(agent, {"approved": True}, "t")
    assert "Draft 2" in (tmp_path / "example-ltd-101.md").read_text(encoding="utf-8")


def test_rejection_without_feedback_and_non_dict_answers(tmp_path: Path) -> None:
    agent = _agent(_search_then_draft(AIMessage("OK.")), ["Draft 1"], tmp_path)
    ask(agent, "cover letter please", "t")
    messages = resume(agent, "no", "t")  # type: ignore[arg-type]
    assert str(messages[0].content).endswith("Feedback: none given")
    assert not list(tmp_path.iterdir())


def test_draft_errors_come_back_as_text(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            call("draft_cover_letter", job_id="999"),
            call("search_jobs", "c2", what="ai"),
            call("draft_cover_letter", "c3", job_id="101"),
            AIMessage("Sorry, drafting failed."),
        ]
    )
    agent = _agent(model, [RuntimeError("429")], tmp_path)
    messages = ask(agent, "cover letter please", "t")
    results = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
    assert results[0].startswith("Error: unknown job id '999'")
    assert results[2] == "Error: drafting failed: RuntimeError"
    assert pending_approval(agent, "t") is None


def test_cover_letter_tool_is_offered_only_with_a_cv_and_not_over_mcp() -> None:
    assert "draft_cover_letter" not in [t.name for t in build_tools(FakeSearch(), Tracker())]
    tools = build_tools(FakeSearch(), Tracker(), writer=LetterWriter(FakeChat([]), CV))
    assert tools[-1].name == "draft_cover_letter"
    server_tools = build_server(tools)._tool_manager.list_tools()
    assert "draft_cover_letter" not in [t.name for t in server_tools]
