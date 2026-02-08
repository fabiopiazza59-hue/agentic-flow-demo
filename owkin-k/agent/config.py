"""
Agent configuration loader.
"""

from pydantic import BaseModel, Field
from typing import Optional
import json
import os


class AgentRuntimeConfig(BaseModel):
    """Configuration for agent runtime."""
    model: str = "claude-sonnet-4-20250514"
    assigned_skills: list[str] = Field(default_factory=lambda: ["pathology-qa"])
    authorized_tools: list[str] = Field(default_factory=lambda: ["pathology.*"])
    max_iterations: int = 5
    token_budget: int = 4000
    gateway_url: str = "http://localhost:8000"
    api_key: str = "ok-demo-key"
    temperature: float = 0.7
    verbose: bool = True


def load_config(config_path: Optional[str] = None) -> AgentRuntimeConfig:
    """Load agent configuration from file or environment."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            data = json.load(f)
            return AgentRuntimeConfig(**data)

    # Default config
    return AgentRuntimeConfig()


# Default configurations for common agent types
AGENT_PRESETS = {
    "pathology-assistant": AgentRuntimeConfig(
        model="claude-sonnet-4-20250514",
        assigned_skills=["pathology-qa"],
        authorized_tools=["pathology.*"],
        max_iterations=5,
        token_budget=4000
    ),
    "model-evaluator": AgentRuntimeConfig(
        model="claude-sonnet-4-20250514",
        assigned_skills=["model-evaluation"],
        authorized_tools=["scoring.*", "datalake.*"],
        max_iterations=8,
        token_budget=6000
    ),
    "report-generator": AgentRuntimeConfig(
        model="claude-sonnet-4-20250514",
        assigned_skills=["report-generation"],
        authorized_tools=["pathology.*", "datalake.*"],
        max_iterations=10,
        token_budget=8000
    ),
    "full-access": AgentRuntimeConfig(
        model="claude-sonnet-4-20250514",
        assigned_skills=["pathology-qa", "model-evaluation", "report-generation"],
        authorized_tools=["pathology.*", "scoring.*", "datalake.*"],
        max_iterations=10,
        token_budget=10000
    )
}
