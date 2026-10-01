"""Tests for the MCP server: the agent's tools, reached through the MCP API."""

import asyncio
import json

import pytest
from langchain_core.tools import BaseTool
from mcp.types import TextContent

from jobagent.mcp_server import build_server
from jobagent.tools import build_tools
from jobagent.tracker import Tracker
from tests.fakes import FakeSearch


def _server() -> tuple[object, Tracker]:
    tracker = Tracker()
    return build_server(build_tools(FakeSearch(), tracker)), tracker


def _text(result: object) -> str:
    block = result.content[0]  # type: ignore[attr-defined]
    assert isinstance(block, TextContent)
    return block.text


def test_lists_the_agent_tools_with_their_schemas() -> None:
    server, _ = _server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}  # type: ignore[attr-defined]
    assert set(tools) == {
        "search_jobs",
        "save_application",
        "update_application",
        "list_applications",
    }
    assert "what" in tools["search_jobs"].input_schema["required"]
    assert "Search live job listings" in (tools["search_jobs"].description or "")


def test_search_then_save_through_mcp_updates_the_tracker() -> None:
    server, tracker = _server()
    found = json.loads(
        _text(asyncio.run(server.call_tool("search_jobs", {"what": "ai"})))  # type: ignore[attr-defined]
    )
    assert found[0]["id"] == "101"
    saved = _text(asyncio.run(server.call_tool("save_application", {"job_id": "101"})))  # type: ignore[attr-defined]
    assert saved.startswith("Saved: Graduate AI Engineer")
    assert tracker.get("101") is not None


def test_rejects_tools_without_a_plain_function() -> None:
    class Opaque(BaseTool):
        name: str = "opaque"
        description: str = "d"

        def _run(self) -> str:
            return "x"

    with pytest.raises(TypeError, match="opaque"):
        build_server([Opaque()])
