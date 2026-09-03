"""
Optional observability: Pydantic AI -> OpenTelemetry -> Arize Phoenix.

Phoenix is the tracing backend already used by `simple-MVP`, so the harness
reports into the same UI (http://localhost:6006). Enable with
`PHOENIX_ENABLED=true`; failures degrade gracefully to "no tracing".
"""

from __future__ import annotations

from .config import Settings


def build_instrumentation(settings: Settings):
    """Return an `InstrumentationSettings` for Pydantic AI, or None when disabled/unavailable."""
    if not settings.phoenix_enabled:
        return None
    try:
        from phoenix.otel import register
        from pydantic_ai import InstrumentationSettings

        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint,
            batch=True,
            set_global_tracer_provider=True,
        )
        print(f"[Tracing] Phoenix initialised -> {settings.phoenix_collector_endpoint}")
        return InstrumentationSettings(tracer_provider=tracer_provider)
    except Exception as exc:  # noqa: BLE001 - tracing must never break the harness
        print(f"[Tracing] Phoenix unavailable, continuing without tracing: {exc}")
        return None
