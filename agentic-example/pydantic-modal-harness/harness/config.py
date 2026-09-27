"""
Harness configuration.

All settings can be provided as environment variables or in a `.env` file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str | None = None
    orchestrator_model: str = "anthropic:claude-opus-5"

    # Compute backend
    modal_mode: Literal["local", "modal"] = "local"
    modal_app_name: str = "agentic-modal-harness"
    local_job_delay_seconds: float = 2.0

    # Result delivery
    harness_public_url: str | None = None
    harness_webhook_secret: str = "change-me-to-a-long-random-string"
    reconcile_interval_seconds: float = 15.0
    reconcile_grace_seconds: float = 20.0
    webhook_max_skew_seconds: int = 300

    # State
    store_backend: Literal["memory", "sqlite"] = "memory"
    sqlite_path: str = "harness.db"

    # Guard rails
    max_run_resumes: int = Field(default=5, description="Max suspend/resume cycles per run")
    max_model_requests_per_run: int = Field(default=12, description="Pydantic AI request_limit")

    # Observability
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "pydantic-modal-harness"

    # Server
    port: int = 8000

    @property
    def callback_url(self) -> str | None:
        """Full webhook URL handed to Modal workers, or None when no public URL is configured."""
        if not self.harness_public_url:
            return None
        return self.harness_public_url.rstrip("/") + "/webhooks/modal"


@lru_cache
def get_settings() -> Settings:
    return Settings()
