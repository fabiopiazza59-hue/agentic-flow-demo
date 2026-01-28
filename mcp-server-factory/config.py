"""
Configuration management for MCP Server Factory.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


def _get_base_path() -> Path:
    """Get the base path for the application."""
    return Path(__file__).parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "MCP Server Factory"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")

    # Paths - using Optional with post-init validation
    base_path: Optional[Path] = Field(default=None, description="Base path for the application")
    registry_data_path: Optional[Path] = Field(default=None, description="Path to registry data directory")
    templates_path: Optional[Path] = Field(default=None, description="Path to Jinja2 templates")
    generated_path: Optional[Path] = Field(default=None, description="Path for generated server files")

    # LLM Settings
    anthropic_api_key: str = Field(default="", description="Anthropic API key for evaluations")
    judge_model: str = Field(
        default="claude-3-5-haiku-latest",
        description="Model to use for LLM-as-judge evaluations",
    )

    # Tracing
    phoenix_endpoint: str = Field(
        default="http://localhost:6006/v1/traces",
        description="Phoenix collector endpoint",
    )
    enable_tracing: bool = Field(default=False, description="Enable Phoenix tracing")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def set_derived_paths(self) -> "Settings":
        """Set derived paths after initialization."""
        if self.base_path is None:
            self.base_path = _get_base_path()
        if self.registry_data_path is None:
            self.registry_data_path = self.base_path / "registry" / "data"
        if self.templates_path is None:
            self.templates_path = self.base_path / "builder" / "templates"
        if self.generated_path is None:
            self.generated_path = self.base_path / "generated" / "servers"

        # Load API key from environment if not set
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def setup_directories(settings: Settings) -> None:
    """Ensure all required directories exist."""
    settings.registry_data_path.mkdir(parents=True, exist_ok=True)
    (settings.registry_data_path / "servers").mkdir(exist_ok=True)
    (settings.registry_data_path / "tools").mkdir(exist_ok=True)
    (settings.registry_data_path / "evaluations").mkdir(exist_ok=True)
    settings.templates_path.mkdir(parents=True, exist_ok=True)
    settings.generated_path.mkdir(parents=True, exist_ok=True)
