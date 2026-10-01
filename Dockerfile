# Runtime image: install deps, then just the agent package itself (no tests,
# evals or dev tooling — those run outside the container, against the repo).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jobagent/ jobagent/

EXPOSE 8501

# data/ (the SQLite tracker + saved letters) and the CV are meant to be
# bind-mounted at runtime, not baked into the image — see docker-compose.yml.
CMD ["streamlit", "run", "jobagent/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
