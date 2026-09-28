# 🧭 Job Hunt Agent

> An AI agent for job seekers in Ireland: it searches real listings, scores
> each one against your CV, drafts tailored cover letters (with your approval),
> and tracks every application.

**Status: in progress.** Built slice by slice; see progress below.

## Progress

- [x] **Slice 1: tools.** Real job search and a SQLite application tracker (see [Job data](#job-data))
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
