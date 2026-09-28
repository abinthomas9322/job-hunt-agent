"""Pick where jobs come from, behind one small interface the tools rely on."""

from typing import Protocol

from jobagent.config import Settings
from jobagent.greenhouse import GreenhouseClient
from jobagent.jobs import Job, JobSearchClient


class JobSource(Protocol):
    """Anything that can search for jobs (Greenhouse boards, Adzuna, a test fake)."""

    def search(
        self, what: str, where: str = "", *, max_days_old: int | None = None, limit: int = 10
    ) -> list[Job]: ...


def build_job_source(settings: Settings) -> JobSource:
    """Create the job source named by ``settings.job_source``."""
    if settings.job_source == "adzuna":
        return JobSearchClient(settings)
    return GreenhouseClient(settings)
