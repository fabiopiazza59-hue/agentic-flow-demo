"""
Pydantic schemas for MCP Server Factory.

Defines the data models for server definitions, tools, and evaluations.
"""

from datetime import datetime
from typing import Literal, Optional, Any

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str = Field(..., description="Parameter name")
    type: Literal["string", "number", "integer", "boolean", "object", "array"] = Field(
        default="string", description="Parameter type"
    )
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[str | int | float | bool] = Field(
        default=None, description="Default value if not provided"
    )


class ToolImplementation(BaseModel):
    """Implementation details for a tool."""

    type: Literal["python", "http", "openapi"] = Field(
        ..., description="Implementation type"
    )
    handler: Optional[str] = Field(
        default=None, description="For python: module.path:function"
    )
    endpoint: Optional[str] = Field(default=None, description="For http: URL endpoint")
    method: str = Field(default="POST", description="For http: HTTP method")


class ToolDefinition(BaseModel):
    """Complete tool definition."""

    id: str = Field(
        ...,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Tool identifier (lowercase, underscores)",
    )
    name: str = Field(..., description="Human-readable tool name")
    description: str = Field(
        ..., min_length=10, description="Tool description for LLM context"
    )
    parameters: list[ToolParameter] = Field(
        default_factory=list, description="Tool parameters"
    )
    implementation: ToolImplementation = Field(..., description="Implementation details")


class ServerConfig(BaseModel):
    """Server configuration options."""

    transport: Literal["stdio", "http"] = Field(
        default="stdio", description="MCP transport type"
    )
    host: str = Field(default="127.0.0.1", description="Host for HTTP transport")
    port: int = Field(default=8001, description="Port for HTTP transport")


class ServerDefinition(BaseModel):
    """Complete MCP server definition."""

    id: str = Field(
        ...,
        pattern=r"^[a-z_][a-z0-9_]*$",
        description="Server identifier (lowercase, underscores)",
    )
    name: str = Field(..., description="Human-readable server name")
    version: str = Field(default="1.0.0", description="Server version")
    description: str = Field(..., description="Server description")
    source_type: Literal["template", "openapi", "custom"] = Field(
        default="template", description="How the server was created"
    )
    config: ServerConfig = Field(
        default_factory=ServerConfig, description="Server configuration"
    )
    tools: list[ToolDefinition] = Field(
        default_factory=list, description="Tools exposed by server"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


class EvaluationResult(BaseModel):
    """Result of a tool evaluation run."""

    id: str = Field(..., description="Evaluation run ID")
    server_id: str = Field(..., description="Server that was evaluated")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Evaluation timestamp"
    )
    tool_selection_accuracy: float = Field(
        ..., ge=0, le=1, description="Accuracy of LLM tool selection"
    )
    param_extraction_accuracy: float = Field(
        ..., ge=0, le=1, description="Accuracy of parameter extraction"
    )
    avg_quality_score: float = Field(
        ..., ge=1, le=5, description="Average quality score (1-5 scale)"
    )
    latency_p50_ms: float = Field(..., ge=0, description="Median latency in ms")
    latency_p95_ms: float = Field(..., ge=0, description="95th percentile latency in ms")
    total_cost_usd: float = Field(..., ge=0, description="Total evaluation cost in USD")
    recommendation: Literal["KEEP", "IMPROVE", "DEPRECATE"] = Field(
        ..., description="Recommended action based on evaluation"
    )
    reasoning: str = Field(..., description="Explanation for recommendation")
    test_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Individual test case results"
    )


class EvaluationSummary(BaseModel):
    """Summary of multiple evaluations for a server."""

    server_id: str
    total_evaluations: int
    avg_quality_score: float
    trend: Literal["improving", "stable", "declining"]
    latest_recommendation: Literal["KEEP", "IMPROVE", "DEPRECATE"]
    latest_evaluation_id: str
