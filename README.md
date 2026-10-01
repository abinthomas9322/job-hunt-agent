# 🧭 Job Hunt Agent

> An AI agent for job seekers in Ireland: it searches real listings, scores
> each one against your CV, drafts tailored cover letters (with your approval),
> and tracks every application.

**Status: in progress.** Built slice by slice; see progress below.

## Progress

- [x] **Slice 1: tools.** Real job search and a SQLite application tracker (see [Job data](#job-data))
- [x] **Slice 2: agent loop.** LangGraph graph where the LLM (Groq) chooses tools, with per-conversation memory and a step limit
- [x] **Slice 3: CV match scoring.** `score_jobs` rates each job 0-100 against your CV with matched/missing skills, using schema-validated structured output; contact details are redacted before the CV reaches the LLM
- [x] **MCP server.** The same tools exposed over the Model Context Protocol, so Claude Desktop, Claude Code or any MCP client can search, score and track jobs (see [Use it from an MCP client](#use-it-from-an-mcp-client))
- [x] **Slice 4: cover letters with human approval.** `draft_cover_letter` writes a letter from CV facts only, then pauses the graph (LangGraph `interrupt`) until you approve, edit or reject it; only approved letters are saved to `data/letters/`, and rejection feedback goes back to the agent for a redraft
- [x] **Slice 5: agent evaluation.** `evals/` runs hand-labelled cases against the
  real LLM (not the scripted fakes `tests/` uses) and checks tool-choice
  accuracy and CV-score accuracy/ranking; see [Evaluating the agent](#evaluating-the-agent)
- [ ] Slice 6: UI, Docker

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                 # add GROQ_API_KEY and CV_PATH
pytest                               # tests + 100% coverage gate
python -m jobagent.cli --trace       # chat with the agent (needs keys in .env)
```

## Job data

- **Greenhouse (default, no key):** Irish postings from the public job boards of
  ten tech employers with Dublin offices (Stripe, Intercom, MongoDB, Datadog,
  Toast, Okta, Twilio, Tines, Flipdish, Dropbox): ~245 Irish jobs on
  2026-09-28, each with its full description. Configure with `GREENHOUSE_BOARDS`.
- **Adzuna (optional, free key):** broad aggregator, but it does **not** cover
  Ireland (the `ie` endpoint returns 404), so it is useful for UK/EU searches
  via `JOB_SOURCE=adzuna` and `ADZUNA_COUNTRY=gb`.

Limitation: company boards list mostly experienced roles; graduate
programmes are often advertised elsewhere (e.g. gradireland), which this
does not cover yet.

## Use it from an MCP client

`jobagent/mcp_server.py` registers every agent tool on an MCP server, reusing
each tool's typed signature and docstring as its schema, so the LangGraph agent
and MCP clients share one contract. Run it over stdio:

```bash
python -m jobagent.mcp_server
```

Example Claude Desktop / Claude Code config (`mcpServers`):

```json
{
  "job-hunt-agent": {
    "command": "/path/to/job-hunt-agent/.venv/bin/python",
    "args": ["-m", "jobagent.mcp_server"],
    "cwd": "/path/to/job-hunt-agent"
  }
}
```

Then ask the client things like "find graduate AI jobs in Dublin and rank them
against my CV". `score_jobs` appears only when `CV_PATH` is set.

## How the agent works

```
START ─► agent (LLM) ──tool calls?──► tools ─┐
             ▲   │                            │
             │   └── plain answer ──► END     │
             └────────── tool results ◄───────┘
```

The LLM sees the conversation plus four tools (`search_jobs`,
`save_application`, `update_application`, `list_applications`) and decides
which to call, in what order, and when it has enough to answer. Tool errors
come back as text so the agent can recover, and a step limit stops runaway
loops. Tests drive the graph with a scripted fake LLM, so they are fast,
free and deterministic.

## Human in the loop

```
agent ─► draft_cover_letter ─► interrupt ── waits ──► you: approve / edit / reject
                                                         │
          ◄── "saved to data/letters/..." or feedback ◄──┘   (Command(resume=...))
```

The graph is checkpointed, so a paused run survives while it waits for you.
Drafts are cached per job, so the letter you approve is exactly the one saved.
In the CLI the draft is shown with a `[y]es / [n]o / [e]dit` prompt. The tool is
not exposed over MCP, because the pause only works inside the agent.

## Evaluating the agent

`tests/` pins the LLM's replies with a scripted fake, so it's fast and
deterministic but can't catch a prompt or tool-description change that makes
the *real* model stop choosing the right tool, or stop scoring jobs sensibly.
`evals/` covers that gap by running hand-labelled cases against the real Groq
model and your real CV:

```bash
python -m evals
```

It needs `GROQ_API_KEY` (and `CV_PATH` for the score cases) in `.env` — real
API calls, so it's not part of `pytest` or CI. It checks:

- **Tool-choice accuracy:** for each of 8 labelled prompts (e.g. "find AI jobs
  in Dublin" → `search_jobs`, a greeting → no tool), does the model ask to call
  the expected tool, or none?
- **Score accuracy:** does `score_jobs` land each hand-picked job in a sane
  0-100 band for your CV (band checks), and does a clearly-better-fit job
  always outscore a clearly-worse one (ranking checks, more robust than an
  exact band since an LLM judge's exact number isn't reproducible to the point)?

Exits non-zero if anything fails, so it also works as a manual gate before
changing `SYSTEM_PROMPT`, a tool's docstring, or the scoring rubric.

## License

MIT
