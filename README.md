# 🧭 Job Hunt Agent

> An AI agent for job seekers in Ireland: it searches real listings, scores
> each one against your CV, drafts tailored cover letters (with your approval),
> and tracks every application.

**Status: in progress.** Built slice by slice; see progress below.

## Progress

- [x] **Slice 1: tools.** Real job search via the [Adzuna API](https://developer.adzuna.com) and a SQLite application tracker
- [x] **Slice 2: agent loop.** LangGraph graph where the LLM (Groq) chooses tools, with per-conversation memory and a step limit
- [x] **Slice 3: CV match scoring.** `score_jobs` rates each job 0-100 against your CV with matched/missing skills, using schema-validated structured output; contact details are redacted before the CV reaches the LLM
- [ ] Slice 4: cover-letter drafts with human approval
- [ ] Slice 5: agent evaluation (tool-choice accuracy, score accuracy)
- [ ] Slice 6: MCP server, UI, Docker

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                 # add your free Adzuna app_id / app_key
pytest                               # tests + 100% coverage gate
python -m jobagent.cli --trace       # chat with the agent (needs keys in .env)
```

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

## License

MIT
