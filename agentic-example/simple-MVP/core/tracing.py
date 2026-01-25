"""
Phoenix Tracing Setup for Scalp MVP

Initializes OpenTelemetry tracing with Arize Phoenix for full observability
of the agentic workflow including:
- LangGraph orchestrator spans
- Agent execution traces
- Tool/MCP call details
"""

import os
from dotenv import load_dotenv

load_dotenv()


def setup_tracing(project_name: str = "scalp-mvp"):
    """
    Initialize Phoenix tracing with auto-instrumentation for LangChain/LangGraph.

    Args:
        project_name: Name of the project in Phoenix UI

    Returns:
        TracerProvider instance
    """
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # Register tracer with Phoenix (uses PHOENIX_COLLECTOR_ENDPOINT env var automatically)
    # Phoenix running locally listens on port 6006 for OTLP/HTTP at /v1/traces
    tracer_provider = register(
        project_name=project_name,
        endpoint="http://localhost:6006/v1/traces",
    )

    # Instrument LangChain (covers LangGraph automatically)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[Tracing] Phoenix initialized - View traces at: http://localhost:6006")

    return tracer_provider


def get_tracer(name: str = "scalp-mvp"):
    """Get a tracer for manual span creation."""
    from opentelemetry import trace
    return trace.get_tracer(name)
