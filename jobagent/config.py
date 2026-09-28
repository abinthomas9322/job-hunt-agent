"""Typed configuration loaded from environment variables / .env.

Every tunable value lives here so no other module reads ``os.environ``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the agent."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Adzuna job search API (free key: https://developer.adzuna.com) ---
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    adzuna_country: str = "ie"
    adzuna_base_url: str = "https://api.adzuna.com/v1/api/jobs"

    # --- LLM (Groq, via its OpenAI-compatible API; free key: console.groq.com) ---
    groq_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    # Hard cap on graph steps per request so a confused agent can't loop forever.
    max_steps: int = 12

    # --- Your CV (.docx, .txt or .md); contact details are redacted on load ---
    cv_path: str = ""

    # --- Application tracker (SQLite file) ---
    db_path: str = "data/applications.db"


def get_settings() -> Settings:
    """Build a Settings instance, reading the environment at call time."""
    return Settings()
