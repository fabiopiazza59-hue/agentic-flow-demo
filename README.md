# Agentic Flow Demo

A complete demonstration of **agentic AI architecture** using LangGraph, Phoenix tracing, and Claude models.

## What's Here

| Folder | Description |
|--------|-------------|
| [`simple-MVP/`](./simple-MVP/) | **Working MVP** - Scalp Trading Assistant with full agentic workflow |
| [`pydantic-modal-harness/`](./pydantic-modal-harness/) | **Async harness** - Pydantic AI orchestrator that suspends on long-running Modal jobs (CPU/GPU/Sandbox) and resumes on callback |
| [`self-improvement_loop/`](./self-improvement_loop/) | **Self-improving forecaster** - daily AMZN close A/B test (LLM ensemble vs. statistical prior + bounded LLM), scored against free baselines with anytime-valid verdicts; runs on GitHub Actions |
| [`trader-project/`](./trader-project/) | **Seamaster** - spec-driven six-stage trade advisor funnel (market scan → triage → risk checks → action table) |
| [`agentic-example/`](./agentic-example/) | Earlier snapshot of this root demo, carried over from the `spec-driven-project` branch |
| [`project-brief.md`](./project-brief.md) | **Full Project Brief** - FinAdvisor AI platform design (1,400+ lines) |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/fabiopiazza59-hue/agentic-flow-demo.git
cd agentic-flow-demo

# Set up the MVP
cd simple-MVP
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and FINNHUB_API_KEY

# Start Phoenix (tracing)
python -m phoenix.server.main serve &

# Start the server
python main.py

# Open the demo UI
open http://localhost:8000
```

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│     ORCHESTRATOR (LangGraph)        │
│     - Intent classification         │
│     - Route to specialist           │
└─────────────────┬───────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌────────┐
│ Scalp  │   │Fallback│   │ More   │
│ Agent  │   │ Agent  │   │Agents..│
│(Sonnet)│   │(Haiku) │   │        │
└────────┘   └────────┘   └────────┘
    │             │
    └─────────────┼─────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         PHOENIX TRACING             │
│    All spans visible at :6006       │
└─────────────────────────────────────┘
```

## Key Patterns Demonstrated

| Pattern | Implementation | File |
|---------|----------------|------|
| **Supervisor Pattern** | LangGraph StateGraph | `orchestrator.py` |
| **ReAct Agents** | create_react_agent | `agents/scalp_agent.py` |
| **Fallback Agent** | Haiku for general queries | `agents/fallback_agent.py` |
| **Model Tiering** | Haiku (routing) + Sonnet (analysis) | Throughout |
| **Tool Integration** | @tool decorator | `mcp_servers/market_data.py` |
| **Phoenix Tracing** | OpenTelemetry auto-instrumentation | `core/tracing.py` |

## Cost Per Query

| Query Type | Model | Cost |
|------------|-------|------|
| Simple/General | Haiku | ~$0.003 |
| Complex Analysis | Sonnet | ~$0.012 |

## Tech Stack

- **Orchestration**: LangGraph
- **LLM**: Claude (Haiku + Sonnet)
- **Tracing**: Arize Phoenix
- **API**: FastAPI
- **Market Data**: Finnhub (free tier)

## Documentation

- [Simple MVP README](./simple-MVP/README.md) - Setup and usage
- [Pydantic AI + Modal Harness](./pydantic-modal-harness/README.md) - Async orchestration with deferred tools, webhook + poll delivery
- [Self-improvement loop](./self-improvement_loop/README.md) - Daily forecaster, [live results](./self-improvement_loop/RESULTS.md), [v2 design](./self-improvement_loop/spec/v2-sota-upgrade.md)
- [Knowledge Base](./simple-MVP/KNOWLEDGE_BASE.md) - Technical reference
- [Scaling Guide](./simple-MVP/docs/SCALING_GUIDE.md) - How to add agents
- [Project Brief](./project-brief.md) - Full FinAdvisor design

## License

MIT
