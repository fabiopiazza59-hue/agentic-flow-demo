"""
Import API routes for legacy tool integration.

Import tools from OpenAPI specs or Python modules.
"""

from fastapi import APIRouter, HTTPException

from registry.store import ServerStore
from adapters.openapi_adapter import OpenAPIAdapter
from adapters.python_adapter import PythonAdapter
from api.models import ImportOpenAPIRequest, ImportPythonRequest, ServerInfo

router = APIRouter(prefix="/import", tags=["import"])

# Dependencies injected from main.py
store: ServerStore = None
openapi_adapter: OpenAPIAdapter = None
python_adapter: PythonAdapter = None


def init_router(
    server_store: ServerStore,
    oa_adapter: OpenAPIAdapter,
    py_adapter: PythonAdapter,
):
    """Initialize router with dependencies."""
    global store, openapi_adapter, python_adapter
    store = server_store
    openapi_adapter = oa_adapter
    python_adapter = py_adapter


@router.post("/openapi", response_model=ServerInfo, status_code=201)
async def import_from_openapi(request: ImportOpenAPIRequest):
    """
    Import MCP server from an OpenAPI specification.

    Fetches the OpenAPI spec from the provided URL and creates
    an MCP server with tools for each endpoint.
    """
    # Check if server already exists
    if store.exists(request.server_id):
        raise HTTPException(
            status_code=409, detail=f"Server '{request.server_id}' already exists"
        )

    try:
        server = await openapi_adapter.import_from_url(
            spec_url=request.spec_url,
            server_id=request.server_id,
            server_name=request.server_name,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
        )

        # Add tags
        server.tags.extend(request.tags)

        # Save to registry
        store.save(server)

        return ServerInfo.from_definition(server)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to import OpenAPI spec: {str(e)}",
        )


@router.post("/python", response_model=ServerInfo, status_code=201)
async def import_from_python(request: ImportPythonRequest):
    """
    Import MCP server from a Python module.

    Introspects the Python module and creates tools from
    functions that have proper type hints and docstrings.
    """
    # Check if server already exists
    if store.exists(request.server_id):
        raise HTTPException(
            status_code=409, detail=f"Server '{request.server_id}' already exists"
        )

    try:
        server = await python_adapter.import_from_module(
            module_path=request.module_path,
            server_id=request.server_id,
            server_name=request.server_name,
            function_filter=request.function_filter,
        )

        # Add tags
        server.tags.extend(request.tags)

        # Save to registry
        store.save(server)

        return ServerInfo.from_definition(server)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to import Python module: {str(e)}",
        )


@router.post("/openapi/preview")
async def preview_openapi_import(request: ImportOpenAPIRequest):
    """
    Preview what tools would be created from an OpenAPI spec.

    Does not save to registry - for inspection only.
    """
    try:
        server = await openapi_adapter.import_from_url(
            spec_url=request.spec_url,
            server_id=request.server_id,
            server_name=request.server_name,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
        )

        return {
            "server_id": request.server_id,
            "server_name": request.server_name,
            "tool_count": len(server.tools),
            "tools": [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description[:100] + "..."
                    if len(t.description) > 100
                    else t.description,
                    "parameters": [p.name for p in t.parameters],
                    "endpoint": t.implementation.endpoint,
                    "method": t.implementation.method,
                }
                for t in server.tools
            ],
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse OpenAPI spec: {str(e)}",
        )
