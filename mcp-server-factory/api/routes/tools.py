"""
Tools API routes.

Manage individual tool definitions (for standalone tools or adding to servers).
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from registry.schemas import ToolDefinition
from registry.store import ServerStore
from api.models import CreateToolRequest

router = APIRouter(prefix="/tools", tags=["tools"])

# Dependencies injected from main.py
store: ServerStore = None


def init_router(server_store: ServerStore):
    """Initialize router with dependencies."""
    global store
    store = server_store


@router.post("/{server_id}/tools", status_code=201)
async def add_tool_to_server(server_id: str, tool: CreateToolRequest):
    """
    Add a new tool to an existing server.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    # Check if tool already exists
    if any(t.id == tool.id for t in server.tools):
        raise HTTPException(
            status_code=409,
            detail=f"Tool '{tool.id}' already exists in server '{server_id}'",
        )

    # Add tool
    new_tool = ToolDefinition(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        implementation=tool.implementation,
    )
    server.tools.append(new_tool)
    store.save(server)

    return {
        "status": "created",
        "server_id": server_id,
        "tool_id": tool.id,
        "message": f"Tool '{tool.id}' added to server '{server_id}'",
    }


@router.get("/{server_id}/tools/{tool_id}")
async def get_tool(server_id: str, tool_id: str):
    """
    Get a specific tool from a server.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    for tool in server.tools:
        if tool.id == tool_id:
            return tool

    raise HTTPException(
        status_code=404,
        detail=f"Tool '{tool_id}' not found in server '{server_id}'",
    )


@router.patch("/{server_id}/tools/{tool_id}")
async def update_tool(server_id: str, tool_id: str, updates: dict):
    """
    Update a tool in a server.

    Allowed updates: name, description, parameters, implementation
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    for i, tool in enumerate(server.tools):
        if tool.id == tool_id:
            # Update fields
            if "name" in updates:
                tool.name = updates["name"]
            if "description" in updates:
                tool.description = updates["description"]
            if "parameters" in updates:
                from registry.schemas import ToolParameter

                tool.parameters = [ToolParameter(**p) for p in updates["parameters"]]
            if "implementation" in updates:
                from registry.schemas import ToolImplementation

                tool.implementation = ToolImplementation(**updates["implementation"])

            server.tools[i] = tool
            store.save(server)
            return {
                "status": "updated",
                "server_id": server_id,
                "tool_id": tool_id,
                "tool": tool.model_dump(),
            }

    raise HTTPException(
        status_code=404,
        detail=f"Tool '{tool_id}' not found in server '{server_id}'",
    )


@router.delete("/{server_id}/tools/{tool_id}")
async def delete_tool(server_id: str, tool_id: str):
    """
    Remove a tool from a server.
    """
    server = store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    for i, tool in enumerate(server.tools):
        if tool.id == tool_id:
            server.tools.pop(i)
            store.save(server)
            return {
                "status": "deleted",
                "server_id": server_id,
                "tool_id": tool_id,
                "message": f"Tool '{tool_id}' removed from server '{server_id}'",
            }

    raise HTTPException(
        status_code=404,
        detail=f"Tool '{tool_id}' not found in server '{server_id}'",
    )
