# 🧭 Job Hunt Agent

> An agentic AI job-search assistant for Ireland: it searches real listings,
> scores each one against **your** CV, drafts tailored cover letters (with
> your approval before anything is saved), and tracks every application —
> from the terminal, a browser UI, or any MCP client.

[![CI](https://github.com/abinthomas9322/job-hunt-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/abinthomas9322/job-hunt-agent/actions/workflows/ci.yml)
[![Coverage 100%](https://img.shields.io/badge/coverage-100%25-brightgreen)](pyproject.toml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](jobagent/ui.py)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Status: complete.** Built in 6 planned slices, each shipped as its own
tested, reviewed commit — see [Progress](#progress).

## What it does

Point it at your real CV once, and it becomes a tool-calling agent that
decides for itself which action a request needs:

- **Searches** real job listings (Greenhouse boards of Dublin tech employers
  by default, no key needed).
- **Scores** any job 0-100 against your CV, with matched/missing skills and a
  one-line reason — using schema-validated structured output, not free text.
- **Drafts cover letters** from CV facts only, then **pauses and waits for
  you** to approve, edit, or reject before anything is saved.
- **Tracks applications** in a local SQLite tracker (saved → applied →
  interview → offer/rejected).

> **Ask** _"Find graduate AI engineer jobs in Dublin and rank them against my
> CV"_ → the agent searches, scores every result, and reports each job's fit
> score with what's missing — no manual copy-pasting of your CV into a prompt.

## Agent evaluation (measured)

`tests/` pins the LLM's replies with a scripted fake, so it's fast and
deterministic — but it can't catch a prompt or tool-description change that
makes the *real* model stop choosing the right tool, or stop scoring jobs
sensibly. [`evals/`](evals/) covers that gap: hand-labelled cases run against
the live Groq model and a real CV.

| Eval | Result |
|---|---|
| Tool-choice accuracy (8 labelled prompts, one per tool + 2 no-tool cases) | **8/8** |
| Score accuracy — band (4 hand-picked jobs vs. an expected 0-100 range) | **4/4** |
| Score accuracy — ranking (clearly-better-fit job must outscore a worse one) | **2/2** |

A real run: a Graduate AI Engineer / RAG role scored **95**, a Retail
Supervisor role scored **88** (the CV's actual past retail experience), a
Commercial Airline Pilot role and a Senior Java role both scored **5**.

```bash
python -m evals   # needs GROQ_API_KEY + CV_PATH in .env; real API calls, not in CI
```

Exits non-zero on any failure, so it also doubles as a manual gate before
changing `SYSTEM_PROMPT`, a tool's docstring, or the scoring rubric.

## Cost

**Runs effectively free.** Groq's LLM API has a free tier, Greenhouse search
needs no API key at all, and the SQLite tracker needs no hosting. No paid
infrastructure is required to run it.

## Quick start

### Option A: Docker (one command)

> Prerequisites: Docker and a free [Groq API key](https://console.groq.com).

```bash
cp .env.example .env    # fill in GROQ_API_KEY, CV_PATH and CV_HOST_PATH
docker compose up --build
```

Open <http://localhost:8501>. The real CV (from `CV_HOST_PATH`) is
bind-mounted read-only; the SQLite tracker and saved letters live in `./data`
on the host, so both survive a `docker compose down`.

### Option B: Run locally

> Prerequisites: Python 3.12 and a free [Groq API key](https://console.groq.com).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                 # add GROQ_API_KEY and CV_PATH

python -m jobagent.cli --trace       # chat with the agent in the terminal
streamlit run jobagent/ui.py         # ...or in the browser
```

### Quality gates

```bash
pytest                       # tests + 100% coverage gate
ruff check . && ruff format --check .
mypy jobagent tests
bandit -r jobagent
pip-audit -r requirements.txt
python -m evals               # real-LLM eval suite (see above)
```

## CI/CD pipeline

`.github/workflows/ci.yml` runs on every push and pull request, with
least-privilege permissions and concurrency-cancel:

- **Secret scan** — gitleaks
- **Python job** — ruff lint, ruff format check, mypy, bandit, pip-audit,
  pytest (+100% coverage)

`evals/` runs against the live model and isn't a CI gate (see
[Agent evaluation](#agent-evaluation-measured)) — it's a manual check before
changing a prompt, tool docstring, or the scoring rubric.

## Use it from an MCP client

`jobagent/mcp_server.py` registers every agent tool on an MCP server, reusing
each tool's typed signature and docstring as its schema, so the LangGraph
agent and MCP clients share one contract. Run it over stdio:

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

Then ask the client things like "find graduate AI jobs in Dublin and rank
them against my CV". `score_jobs` appears only when `CV_PATH` is set.
`draft_cover_letter` is **not** exposed over MCP — its approval pause only
works inside the agent's own LangGraph run.

## Features

- 🔍 **Real job search** — Greenhouse boards of 10 Dublin tech employers
  (no key), or Adzuna as a UK/EU fallback
- 🎯 **CV-fit scoring** — 0-100, matched/missing skills, a one-line reason;
  contact details are redacted before the CV ever reaches the LLM
- ✍️ **Cover letters with human approval** — a LangGraph `interrupt` pauses
  the graph until you approve, edit, or reject; nothing is saved otherwise
- 📋 **Application tracker** — SQLite-backed, full status workflow
- 🔌 **MCP server** — the same tools, over the Model Context Protocol, for
  Claude Desktop/Code or any other MCP client
- 💬 **Two front-ends, one agent** — terminal CLI or Streamlit browser UI,
  the same LangGraph graph underneath
- 🧪 **Evaluated against the real model** — not just scripted-fake unit tests
  (see [Agent evaluation](#agent-evaluation-measured))
- 🐳 **Dockerised** — runtime-only image; the CV, tracker and letters are
  bind-mounted, never baked in

## Architecture

```
START ─► agent (LLM) ──tool calls?──► tools ─┐
             ▲   │                            │
             │   └── plain answer ──► END     │
             └────────── tool results ◄───────┘
```

The LLM (Groq) sees the conversation plus whatever tools are available and
decides which to call, in what order, and when it has enough to answer. Tool
errors come back as text so the agent can recover instead of crashing, and a
step limit stops runaway loops.

| Tool | Purpose |
|---|---|
| `search_jobs` | Search live listings |
| `save_application` | Start tracking a job |
| `update_application` | Change a tracked job's status |
| `list_applications` | List tracked applications |
| `score_jobs` | CV-fit score + matched/missing skills (only when `CV_PATH` is set) |
| `draft_cover_letter` | Draft + human-approved cover letter (CLI/UI only) |

**Human in the loop:**

```
agent ─► draft_cover_letter ─► interrupt ── waits ──► you: approve / edit / reject
                                                         │
          ◄── "saved to data/letters/..." or feedback ◄──┘   (Command(resume=...))
```

The graph is checkpointed, so a paused run survives while it waits for you.
Drafts are cached per job, so the letter you approve is exactly the one
saved; rejection feedback goes back to the agent for a redraft.

**Two front-ends over one agent:** `jobagent/cli.py` and `jobagent/ui.py`
(Streamlit) are both thin wrappers around the same `ask` /
`pending_approval` / `resume` functions in `jobagent/agent.py` — the
LangGraph checkpointer holds the real conversation state, so the UI only
keeps a small display-only transcript to redraw on each rerun.

## Key technical decisions & why

- **LangGraph `interrupt` for human-in-the-loop** — the graph pauses exactly
  where approval is needed and resumes with `Command(resume=...)`, instead of
  a separate approval state machine bolted on.
- **Pydantic structured output for `score_jobs`** — the score is always a
  validated 0-100 int and the skill lists are real lists, safe to sort, store
  and test, never free text to parse.
- **CV redaction before the model sees it** — emails and phone numbers are
  stripped in `jobagent/cv.py` so contact details never leave the machine via
  the LLM API.
- **One typed function, two consumers** — each tool is a single Python
  function with a typed signature and docstring; both the LangGraph agent and
  the MCP server build their schema from it, so they can't drift apart.
- **Real-model evals, separate from unit tests** — `tests/` pins the LLM's
  replies for fast, free, deterministic CI; `evals/` catches the class of
  regression scripted fakes can't (a prompt change that makes the *real*
  model stop calling the right tool).
- **Streamlit over a leaner FastAPI+HTML UI** — chosen for build speed and
  consistency with my other portfolio project's UI, accepting a heavier
  dependency footprint (pandas/numpy/pyarrow/altair that the agent itself
  never uses) as the trade-off.
- **Groq over a paid LLM provider** — a real tool-calling model with a
  generous free tier, so the whole thing runs at zero cost.

## Productionizing & scaling

The current build is sized for one person running their own job search. To
take it further I'd:

- Add **multi-user auth** and per-user data isolation — today it's one
  `.env`, one CV, one SQLite file for a single local user.
- Move the tracker from **SQLite to Postgres** for concurrent multi-session
  writes.
- Add **more job sources** — graduate-programme boards (e.g. gradireland)
  that the current Greenhouse boards don't cover, and a scheduled
  refresh/scrape instead of search-on-demand only.
- Add **rate limiting and observability** — structured logs and metrics
  around LLM calls and tool execution, beyond the existing retry/backoff.
- The Docker image is ready to deploy, but since it handles a real CV and a
  personal API key, I've kept it local/self-hosted rather than a public
  demo — a real deployment needs per-user secrets and auth first.

## Engineering standards I followed (and skipped)

**Followed:**
- Built in 6 planned slices, one tested, reviewed commit per slice — never
  starting the next until the current one was green.
- **100% test coverage, enforced in CI** (`--cov-fail-under=100`); the two
  interactive shells (`cli.py`, `ui.py`) are explicitly excluded and reasoned
  about directly rather than faked.
- Fully typed, `mypy --disallow-untyped-defs` clean.
- Security scanning from day one — gitleaks, bandit, pip-audit, all clean.
- **Evaluated against the real LLM**, not just scripted fakes — tool-choice
  and scoring accuracy both measured live (14/14), runnable as a manual gate.
- **Human-in-the-loop by construction** — a cover letter can't be saved
  without explicit approval; the agent can't silently take that action.
- The Docker image holds no secrets or user data — the CV and the tracker/
  letters are bind-mounted at runtime, never baked into the image.

**Skipped (knowingly, for now):**
- No CodeQL/Trivy workflow — CI covers gitleaks/bandit/pip-audit, not a
  separate SAST or container scan.
- No auth — single local user by design.
- No Postgres — SQLite is plenty for one person's application tracker.
- No public hosted deployment — it holds a personal CV and API key, so it
  stays local/self-hosted rather than a public demo.

## How I used AI tools in development

> ✍️ _Draft — replace with my own words._
>
> I built this with Claude Code, working in small tested slices and reviewing
> every diff myself. I made the architecture calls — the LangGraph
> `interrupt` pattern for approval, sharing one typed tool between the agent
> and the MCP server, and the Streamlit-vs-FastAPI trade-off for the UI. I
> trusted it less on _[prompt wording / dependency choices / rate-limit
> assumptions — fill in]_ and checked those by hand.

## What I'd do differently with more time

> ✍️ _Draft — replace with my own words._
>
> With more time I'd add real multi-user auth and a hosted deployment, wider
> job-source coverage (graduate programmes, not just experienced-hire
> boards), and a background refresh job instead of search-on-demand only.
> _[Add the things you'd prioritise.]_

## Edge cases knowingly skipped

- Job data comes from 10 Greenhouse boards (~245 Irish jobs) — mostly
  experienced roles; graduate programmes elsewhere aren't covered.
- Adzuna doesn't cover Ireland (the `ie` endpoint 404s) — only useful as a
  UK/EU fallback.
- Job descriptions are truncated to 2,000 characters before scoring or
  cover-letter drafting, to keep one LLM call within free-tier token limits.
- At most 5 jobs are scored per `score_jobs` call, for the same reason.
- Concurrent runs compete for the same Groq subscription quota — don't run
  two generations at once.
- CV must be `.docx`, `.txt` or `.md`; no OCR for scanned/image CVs.

## License

[MIT](LICENSE) © 2026 Abin Oommen Thomas

## About / Topics

A portfolio-grade agentic AI job-search assistant for Ireland: a LangGraph
tool-calling agent with CV-fit scoring, human-approved cover letters, an MCP
server, a Streamlit UI and Docker — Python · LangGraph · LangChain · Groq ·
MCP · Streamlit · Docker.

**Topics:** `langgraph` · `langchain` · `agentic-ai` · `llm` · `groq` ·
`mcp` · `model-context-protocol` · `streamlit` · `docker` · `job-search` ·
`python` · `portfolio`
