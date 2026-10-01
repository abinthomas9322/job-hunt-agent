"""Expose the agent's tools over MCP (Model Context Protocol).

Any MCP client (Claude Desktop, Claude Code, an IDE assistant) can then search
Irish jobs, score them against your CV and track applications: the same tools
the LangGraph agent uses, with no extra code. The LLM is the client's own; this
server only runs the tools (``score_jobs`` still uses the Groq model).

Run it over stdio from the repo root::

    python -m jobagent.mcp_server
"""

from langchain_core.tools import BaseTool, StructuredTool
from mcp.server.mcpserver import MCPServer

from jobagent.config import get_settings
from jobagent.cv import load_cv
from jobagent.matching import Matcher
from jobagent.sources import build_job_source
from jobagent.tools import NEEDS_APPROVAL, build_tools
from jobagent.tracker import Tracker

INSTRUCTIONS = (
    "Tools for an Irish job hunt. Use search_jobs first: other tools take job ids "
    "from its results. score_jobs ranks jobs against the user's CV."
)


def build_server(tools: list[BaseTool]) -> MCPServer:
    """Register each LangChain tool on an MCP server under the same name.

    The tools' typed signatures and docstrings become the MCP input schemas and
    descriptions, so the agent and MCP clients see exactly the same contract.
    Tools that pause for human approval are skipped: the pause only works
    inside the LangGraph agent.
    """
    server = MCPServer("job-hunt-agent", instructions=INSTRUCTIONS)
    for t in tools:
        if NEEDS_APPROVAL in (t.tags or []):
            continue
        if not isinstance(t, StructuredTool) or t.func is None:
            raise TypeError(f"tool {t.name!r} has no plain function to expose")
        server.add_tool(t.func, name=t.name, description=t.description)
    return server


def main() -> None:  # pragma: no cover - wires tested pieces to real services
    settings = get_settings()
    matcher = None
    if settings.cv_path:
        from jobagent.agent import build_llm

        matcher = Matcher(build_llm(settings), load_cv(settings.cv_path))  # type: ignore[arg-type]
    tools = build_tools(build_job_source(settings), Tracker(settings.db_path), matcher)
    build_server(tools).run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
