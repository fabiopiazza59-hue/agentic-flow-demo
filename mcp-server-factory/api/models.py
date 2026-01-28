"""
API request/response models for MCP Server Factory.
"""

from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field

from registry.schemas import (
    ToolParameter,
    ToolImplementation,
    ToolDefinition,
    ServerConfig,
    ServerDefinition,
)


# ============================================================
# SERVER MODELS
# ============================================================


class CreateToolRequest(BaseModel):
    """Request to create a new tool."""

    id: str = Field(
        ..., pattern=r"^[a-z_][a-z0-9_]*$", description="Tool identifier"
    )
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., min_length=10, description="Tool description")
    parameters: list[ToolParameter] = Field(default_factory=list)
    implementation: ToolImplementation


class CreateServerRequest(BaseModel):
    """Request to create a new MCP server."""

    id: str = Field(
        ..., pattern=r"^[a-z_][a-z0-9_]*$", description="Server identifier"
    )
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Server description")
    source_type: str = Field(default="template")
    config: Optional[ServerConfig] = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class UpdateServerRequest(BaseModel):
    """Request to update an existing server."""

    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[ServerConfig] = None
    tools: Optional[list[ToolDefinition]] = None
    tags: Optional[list[str]] = None


class ServerInfo(BaseModel):
    """Summary information about a server."""

    id: str
    name: str
    version: str
    description: str
    source_type: str
    tool_count: int
    tags: list[str]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_definition(cls, server: ServerDefinition) -> "ServerInfo":
        """Create from ServerDefinition."""
        return cls(
            id=server.id,
            name=server.name,
            version=server.version,
            description=server.description,
            source_type=server.source_type,
            tool_count=len(server.tools),
            tags=server.tags,
            created_at=server.created_at,
            updated_at=server.updated_at,
        )


class GeneratedCodeResponse(BaseModel):
    """Response containing generated code."""

    server_id: str
    code: str
    filename: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response after generating server file."""

    server_id: str
    output_path: str
    message: str


# ============================================================
# IMPORT MODELS
# ============================================================


class ImportOpenAPIRequest(BaseModel):
    """Request to import from OpenAPI spec."""

    spec_url: str = Field(..., description="URL to OpenAPI spec JSON")
    server_id: str = Field(
        ..., pattern=r"^[a-z_][a-z0-9_]*$", description="ID for new server"
    )
    server_name: str = Field(..., description="Name for new server")
    include_patterns: Optional[list[str]] = Field(
        default=None, description="Path patterns to include"
    )
    exclude_patterns: Optional[list[str]] = Field(
        default=None, description="Path patterns to exclude"
    )
    tags: list[str] = Field(default_factory=list)


class ImportPythonRequest(BaseModel):
    """Request to import from Python module."""

    module_path: str = Field(
        ..., description="Python module path (e.g., 'my_package.tools')"
    )
    server_id: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    server_name: str
    function_filter: Optional[list[str]] = Field(
        default=None, description="Function names to import (None = all)"
    )
    tags: list[str] = Field(default_factory=list)


# ============================================================
# EVALUATION MODELS
# ============================================================


class RunEvaluationRequest(BaseModel):
    """Request to run evaluation on a server."""

    test_cases: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Custom test cases (auto-generated if not provided)"
    )
    num_synthetic_cases: int = Field(
        default=5, ge=1, le=20, description="Number of synthetic test cases to generate"
    )


class EvaluationSummaryResponse(BaseModel):
    """Summary of an evaluation run."""

    id: str
    server_id: str
    timestamp: datetime
    avg_quality_score: float
    tool_selection_accuracy: float
    recommendation: str
    reasoning: str


# ============================================================
# TEMPLATE MODELS
# ============================================================


class TemplateInfo(BaseModel):
    """Information about a server template."""

    id: str
    name: str
    description: str
    category: str
    tool_count: int
    example_tools: list[str]


class CreateFromTemplateRequest(BaseModel):
    """Request to create server from template."""

    template_id: str
    server_id: str = Field(..., pattern=r"^[a-z_][a-z0-9_]*$")
    server_name: str
    description: Optional[str] = None
    customizations: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
