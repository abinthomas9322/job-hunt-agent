# 🧭 Job Hunt Agent

> An AI agent for job seekers in Ireland: it searches real listings, scores
> each one against your CV, drafts tailored cover letters (with your approval),
> and tracks every application.

**Status: in progress.** Built slice by slice; see progress below.

## Progress

- [x] **Slice 1: tools.** Real job search via the [Adzuna API](https://developer.adzuna.com) and a SQLite application tracker
- [ ] Slice 2: agent loop (LangGraph + tool calling)
- [ ] Slice 3: CV ↔ job match scoring
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
```

## License

MIT
