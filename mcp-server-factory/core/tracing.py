"""
Phoenix Tracing Setup for MCP Server Factory.

Initializes OpenTelemetry tracing with Arize Phoenix for observability.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def setup_tracing(project_name: str = "mcp-server-factory"):
    """
    Initialize Phoenix tracing with auto-instrumentation.

    Args:
        project_name: Name of the project in Phoenix UI

    Returns:
        TracerProvider instance
    """
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Register tracer with Phoenix
        tracer_provider = register(
            project_name=project_name,
            endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        )

        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        print(f"[Tracing] Phoenix initialized - View traces at: http://localhost:6006")

        return tracer_provider

    except ImportError:
        print("[Tracing] Phoenix not installed, tracing disabled")
        return None
    except Exception as e:
        print(f"[Tracing] Failed to initialize: {e}")
        return None


def get_tracer(name: str = "mcp-server-factory"):
    """Get a tracer for manual span creation."""
    from opentelemetry import trace

    return trace.get_tracer(name)
