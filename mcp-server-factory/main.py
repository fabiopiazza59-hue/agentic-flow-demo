"""
MCP Server Factory - FastAPI Entry Point

Self-serve MCP server creation with built-in evaluation.

Run: python main.py
API: http://localhost:8000
Docs: http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

from config import get_settings, setup_directories
from registry.store import ServerStore, EvaluationStore
from builder.generator import ServerGenerator
from adapters.openapi_adapter import OpenAPIAdapter
from adapters.python_adapter import PythonAdapter

# Import route modules
from api.routes import servers, templates, tools, import_legacy, evaluation


# ============================================================
# LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    settings = get_settings()

    print("\n" + "=" * 60)
    print("  MCP SERVER FACTORY")
    print("=" * 60)

    # Ensure directories exist
    setup_directories(settings)
    print("[Setup] Directories initialized")

    # Initialize stores
    server_store = ServerStore(settings.registry_data_path)
    evaluation_store = EvaluationStore(settings.registry_data_path)
    print("[Setup] Stores initialized")

    # Initialize generator
    generator = ServerGenerator(
        templates_path=settings.templates_path,
        output_path=settings.generated_path,
    )
    print("[Setup] Generator initialized")

    # Initialize adapters
    openapi_adapter = OpenAPIAdapter()
    python_adapter = PythonAdapter()
    print("[Setup] Adapters initialized")

    # Initialize route modules with dependencies
    servers.init_router(server_store, generator)
    templates.init_router(server_store)
    tools.init_router(server_store)
    import_legacy.init_router(server_store, openapi_adapter, python_adapter)
    evaluation.init_router(server_store, evaluation_store)
    print("[Setup] Routes initialized")

    # Setup tracing if enabled
    if settings.enable_tracing:
        try:
            from core.tracing import setup_tracing

            setup_tracing("mcp-server-factory")
            print(f"[Tracing] Phoenix: {settings.phoenix_endpoint}")
        except Exception as e:
            print(f"[Tracing] Failed to initialize: {e}")

    print(f"\n[Server] Starting on http://{settings.host}:{settings.port}")
    print(f"[Server] API Docs: http://{settings.host}:{settings.port}/docs")
    print("=" * 60 + "\n")

    yield

    print("\n[Server] Shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="MCP Server Factory",
    description="Self-serve MCP server creation and legacy tool integration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(servers.router)
app.include_router(templates.router)
app.include_router(tools.router)
app.include_router(import_legacy.router)
app.include_router(evaluation.router)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")


# ============================================================
# ROUTES
# ============================================================


@app.get("/")
async def root():
    """Serve the main UI page."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "name": "MCP Server Factory",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "servers": "/servers",
            "templates": "/templates",
            "import": "/import",
            "evaluate": "/evaluate",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "version": settings.app_version,
        "components": {
            "registry": "ready",
            "generator": "ready",
            "adapters": "ready",
            "evaluation": "ready" if settings.anthropic_api_key else "api_key_missing",
        },
        "config": {
            "anthropic_key": "configured" if settings.anthropic_api_key else "missing",
            "tracing": "enabled" if settings.enable_tracing else "disabled",
        },
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
