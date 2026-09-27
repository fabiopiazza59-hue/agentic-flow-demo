"""
Pydantic AI + Modal Harness

An orchestration harness where a Pydantic AI agent can hand long-running
work (GPU inference, large simulations, sandboxed code) to Modal, suspend
its run, and resume automatically once Modal reports back.

Modules:
- config        Settings loaded from the environment / .env
- models        Run / Job / Event data models
- store         Durable state (in-memory or SQLite)
- events        Per-run pub/sub used by the SSE endpoint
- security      HMAC signing + verification for webhooks
- workloads     Pure-Python compute shared by Modal and local mode
- dispatch      Job backends (Modal or in-process)
- jobs          Job lifecycle: dispatch, result handling, reconciliation
- orchestrator  The Pydantic AI agent and the suspend/resume loop
- tracing       Optional Phoenix / OpenTelemetry instrumentation
"""
