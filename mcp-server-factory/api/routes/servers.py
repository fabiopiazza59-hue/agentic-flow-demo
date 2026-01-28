"""
Server management API routes.

CRUD operations for MCP server definitions.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from registry.schemas import ServerDefinition
from registry.store import ServerStore
from builder.generator import ServerGenerator
from api.models import (
    CreateServerRequest,
    UpdateServerRequest,
    ServerInfo,
    GeneratedCodeResponse,
    GenerateResponse,
)

router = APIRouter(prefix="/servers", tags=["servers"])

# Dependencies injected from main.py
store: ServerStore = None
generator: ServerGenerator = None


def init_router(server_store: ServerStore, server_generator: ServerGenerator):
    """Initialize router with dependencies."""
    global store, generator
    store = server_store
    generator = server_generator


@router.get("", response_model=list[ServerInfo])
async def list_servers(
    tags: Optional[list[str]] = Query(default=None, description="Filter by tags")
):
    """
    List all registered servers.

    Optionally filter by tags (OR logic - matches any tag).
    """
    servers = store.list_all(tags=tags)
    return [ServerInfo.from_definition(s) for s in servers]


@router.post("", response_model=ServerInfo, status_code=201)
async def create_server(request: CreateServerRequest):
    """
    Create a new MCP server definition.

    The server is saved to the registry and can then be used to
    generate Python code.
    """
    # Check if server already exists
    if store.exists(request.id):
        raise HTTPException(
            status_code=409, detail=f"Server '{request.id}' already exists"
        )

    # Build server definition, using defaults for optional fields
    server_data = {
        "id": request.id,
        "name": request.name,
        "description": request.description,
        "source_type": request.source_type,
        "tools": request.tools,
        "tags": request.tags,
    }
    # Only include config if provided
    if request.config is not None:
        server_data["config"] = request.config

    server = ServerDefinition(**server_data)
    store.save(server)
    return ServerInfo.from_definition(server)


@router.get("/{server_id}", response_model=ServerDefinition)
async def get_server(server_id: str):
    """
    Get full server definition by ID.

    Returns the complete server configuration including all tools.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")
    return server


@router.patch("/{server_id}", response_model=ServerInfo)
async def update_server(server_id: str, request: UpdateServerRequest):
    """
    Update an existing server definition.

    Only provided fields will be updated.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    # Update only provided fields
    if request.name is not None:
        server.name = request.name
    if request.description is not None:
        server.description = request.description
    if request.config is not None:
        server.config = request.config
    if request.tools is not None:
        server.tools = request.tools
    if request.tags is not None:
        server.tags = request.tags

    store.save(server)
    return ServerInfo.from_definition(server)


@router.delete("/{server_id}")
async def delete_server(server_id: str):
    """
    Archive a server.

    The server is moved to the archive directory rather than being
    permanently deleted.
    """
    if store.delete(server_id):
        return {"status": "archived", "server_id": server_id}
    raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")


@router.post("/{server_id}/preview", response_model=GeneratedCodeResponse)
async def preview_code(server_id: str):
    """
    Preview the generated Python code for a server.

    Returns the code without saving it to disk.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    code = generator.preview(server)
    return GeneratedCodeResponse(
        server_id=server_id,
        code=code,
        filename=f"{server_id}_server.py",
    )


@router.post("/{server_id}/generate", response_model=GenerateResponse)
async def generate_server(server_id: str):
    """
    Generate the Python server file.

    Creates a runnable MCP server in the generated/ directory.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    output_path = generator.generate(server)
    return GenerateResponse(
        server_id=server_id,
        output_path=str(output_path),
        message=f"Server generated at {output_path}",
    )


@router.get("/{server_id}/tools")
async def list_server_tools(server_id: str):
    """
    List all tools for a specific server.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    return {
        "server_id": server_id,
        "tool_count": len(server.tools),
        "tools": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description[:100] + "..."
                if len(t.description) > 100
                else t.description,
                "parameter_count": len(t.parameters),
                "implementation_type": t.implementation.type,
            }
            for t in server.tools
        ],
    }
