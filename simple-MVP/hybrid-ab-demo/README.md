# Hybrid AI Demo - ReAct vs CodeAct A/B Testing

A minimal showcase demonstrating hybrid agentic architecture with A/B testing capabilities.

## Architecture

```
Query → Classifier (complexity analysis)
              │
    ┌─────────┴─────────┐
    │                   │
Complexity ≤3       Complexity ≥4
    │                   │
    ▼                   ▼
 ReAct              CodeAct
 (Haiku)            (Sonnet)
    │                   │
 MCP Tools        Python Sandbox
    │                   │
    └─────────┬─────────┘
              ▼
      Response + Metrics
              │
              ▼
         Phoenix Traces
```

## Features

- **Hybrid Routing**: Auto-routes simple queries to ReAct (tools), complex to CodeAct (Python)
- **A/B Testing**: Run same query through BOTH paths, compare side-by-side
- **Metrics**: Latency, cost, accuracy (LLM-as-judge evaluation)
- **Phoenix Observability**: Full tracing with custom A/B attributes

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your ANTHROPIC_API_KEY

# 2. Start Phoenix (optional, for traces)
python -m phoenix.server.main serve

# 3. Run server
python main_hybrid.py

# 4. Open UI
open http://localhost:8000
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Demo UI with tabs for single query and A/B comparison |
| `POST /analyze` | Auto-routed query (ReAct or CodeAct) |
| `POST /ab-test` | Run query through BOTH paths, return comparison |
| `GET /health` | Health check |

## A/B Test Response

```json
{
  "query": "Calculate compound interest...",
  "react": {
    "latency_ms": 5500,
    "cost_usd": 0.0005,
    "evaluation": {"overall": 4.75}
  },
  "codeact": {
    "latency_ms": 6500,
    "cost_usd": 0.0070,
    "evaluation": {"overall": 5.0},
    "code": "# Python code..."
  },
  "winner": "react",
  "comparison": {
    "cost_ratio": 14.0,
    "latency_diff_ms": 1000
  }
}
```

## Phoenix Trace Attributes

Filter traces by:
- `ab.path` → "react" or "codeact"
- `ab.winner` → which path won
- `ab.cost_usd` → cost comparison
- `ab.latency_ms` → timing
- `ab.react_score` / `ab.codeact_score` → quality scores

## File Structure

```
hybrid-ab-demo/
├── main_hybrid.py           # FastAPI server
├── hybrid_orchestrator.py   # LangGraph routing
├── ab_testing.py            # A/B test framework
├── agents/
│   ├── react_agent.py       # Tool-based agent (Haiku)
│   └── codeact_agent.py     # Code generation agent (Sonnet)
├── code_interpreter/
│   └── sandbox.py           # Safe Python execution
├── core/
│   └── tracing.py           # Phoenix setup
├── mcp_servers/
│   └── market_data.py       # Finnhub tools
└── static/
    └── hybrid_index.html    # Demo UI
```

## Key Findings

| Metric | ReAct | CodeAct |
|--------|-------|---------|
| Cost | ~$0.0005/query | ~$0.007/query (14x more) |
| Latency | ~5-8s | ~6-10s |
| Accuracy | ~4.0/5 | ~4.5/5 (slightly better) |

**Recommendation**: Use ReAct for cost-sensitive, simple queries. Use CodeAct when accuracy on complex computations is critical.
