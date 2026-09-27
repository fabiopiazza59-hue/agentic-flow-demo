# Scalp Trading Assistant - Agentic MVP

A minimal but complete agentic AI system demonstrating the supervisor pattern with LangGraph, real-time market data, and full observability through Phoenix tracing.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Components Deep Dive](#components-deep-dive)
- [Setup & Installation](#setup--installation)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [V2.1 Methodology](#v21-methodology)
- [Extending the System](#extending-the-system)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
│     "Analyze NVDA..." or "What is RSI?" or "Hello"              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (LangGraph)                      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Classifier  │───▶│    Router    │───▶│  Responder   │      │
│  │   (Haiku)    │    │              │    │              │      │
│  └──────────────┘    └──────┬───────┘    └──────────────┘      │
│                             │                                    │
│         SCALP_ANALYSIS ─────┼───── GENERAL                      │
│                             │                                    │
└─────────────────────────────┼───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────────┐
│     SCALP AGENT (ReAct)     │ │     FALLBACK AGENT (ReAct)      │
│       Claude Sonnet         │ │        Claude Haiku             │
│                             │ │                                 │
│  Tools:                     │ │  Tools:                         │
│  - get_stock_quote          │ │  - get_system_info              │
│  - get_spy_change           │ │  - get_trading_terminology      │
│  - calculate_confluence     │ │  - calculate_basic_math         │
└─────────────────────────────┘ └─────────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHOENIX TRACING                               │
│              All spans captured for observability                │
│                   http://localhost:6006                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration | LangGraph StateGraph | Type-safe state management, conditional routing |
| Classifier LLM | Claude Haiku | Cost-efficient ($0.001/call), fast routing |
| Scalp Agent LLM | Claude Sonnet | Better reasoning for complex analysis ($0.01/call) |
| Fallback Agent LLM | Claude Haiku | Cost-efficient for simple queries ($0.001/call) |
| Agent Pattern | ReAct | Reasoning + Acting loop with tool calls |
| Market Data | Finnhub API | Free tier, real-time quotes |
| Observability | Arize Phoenix | Open-source, self-hosted, full trace visibility |

---

## Components Deep Dive

### 1. Orchestrator (`orchestrator.py`)

The orchestrator implements the **supervisor pattern** using LangGraph's StateGraph:

```python
# State flows through the graph
class OrchestratorState(TypedDict):
    messages: Annotated[list, operator.add]  # Accumulated messages
    intent: str                               # Classified intent
    analysis_result: dict | None              # Final result
```

**Nodes:**
- `classifier` - Uses Haiku to classify intent (SCALP_ANALYSIS, GENERAL, UNKNOWN)
- `scalp_agent` - Invokes the ReAct agent for trading analysis
- `general_responder` - Handles non-trading queries
- `format_response` - Prepares final output

**Flow:**
```
START → classifier → [scalp_agent | general_responder] → format_response → END
```

**Key Code:**
```python
workflow.add_conditional_edges(
    "classifier",
    route_to_agent,
    {
        "scalp_agent": "scalp_agent",
        "general_responder": "general_responder"
    }
)
```

### 2. Scalp Agent (`agents/scalp_agent.py`)

A **ReAct agent** that reasons about market data and decides on trade setups:

```python
agent = create_react_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    tools=[get_stock_quote, get_spy_change, calculate_confluence],
    prompt=SCALP_AGENT_PROMPT,  # V2.1 methodology instructions
)
```

**ReAct Loop:**
1. **Reason** - Agent thinks about what data it needs
2. **Act** - Calls tools (get_stock_quote, get_spy_change)
3. **Observe** - Receives tool results
4. **Repeat** - Until it has enough info
5. **Respond** - Calls calculate_confluence, formats final answer

### 3. Market Data Tools (`mcp_servers/market_data.py`)

Two tools that fetch real-time data from Finnhub:

```python
@tool
def get_stock_quote(symbol: str) -> dict:
    """Get current stock quote from Finnhub."""
    # Returns: {symbol, price, high, low, change_percent, timestamp}

@tool
def get_spy_change() -> dict:
    """Get SPY % change for market context."""
    # Returns: {symbol, change_percent, price, market_condition}
```

**Finnhub API:**
- Free tier: 60 calls/minute
- Endpoint: `https://finnhub.io/api/v1/quote`
- Returns: current price (c), high (h), low (l), open (o), previous close (pc)

### 4. Confluence Calculator (`skills/confluence.py`)

Implements the **V2.1 scoring methodology**:

```python
@tool
def calculate_confluence(
    symbol: str,
    price: float,
    high: float,
    low: float,
    rsi: int,
    volume_ratio: float,
    vwap_position: str,    # "above" | "below" | "touching"
    trend: str,            # "up" | "down" | "sideways"
    support: float,
    spy_change: float,
    has_catalyst: bool = False
) -> dict:
    """Calculate V2.1 confluence score and trade decision."""
```

**Returns:**
```python
{
    "symbol": "NVDA",
    "score": 7.5,
    "max_score": 10,
    "verdict": "LONG_SCALP",
    "confidence": "HIGH",
    "factors": {...},
    "trade_setup": {
        "entry": 187.68,
        "stop_loss": 185.26,
        "target_1": 189.44,
        "target_2": 191.04,
        "risk_reward": "3.0:1"
    },
    "reasoning": "..."
}
```

### 5. Phoenix Tracing (`core/tracing.py`)

Sets up OpenTelemetry tracing with auto-instrumentation:

```python
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

def setup_tracing(project_name: str = "scalp-mvp"):
    tracer_provider = register(
        project_name=project_name,
        endpoint="http://localhost:6006/v1/traces",
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

**What's Traced:**
- LangGraph node transitions
- LLM calls (input/output tokens, latency)
- Tool invocations (arguments, results)
- Agent reasoning steps

### 6. FastAPI Server (`main.py`)

Entry point with single `/analyze` endpoint:

```python
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    orchestrator = get_orchestrator()
    result = await orchestrator.ainvoke({
        "messages": [HumanMessage(content=request.query)],
        "intent": "",
        "analysis_result": None
    })
    return AnalyzeResponse(...)
```

### 7. Demo UI (`static/index.html`)

Single-page web interface:
- Dark gradient theme
- Query input with examples
- Real-time flow visualization
- Formatted response display
- Phoenix link for traces

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Anthropic API key (with credits)
- Finnhub API key (free at finnhub.io)

### Quick Start

```bash
# 1. Navigate to project
cd agentic-example/simple-MVP

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Start Phoenix (in separate terminal)
python -m phoenix.server.main serve

# 6. Start the server
python main.py

# 7. Open UI
open http://localhost:8000
```

### Environment Variables

```env
# Required
ANTHROPIC_API_KEY=sk-ant-api03-...

# Required for real market data
FINNHUB_API_KEY=your-finnhub-key

# Optional (defaults shown)
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
PORT=8000
```

### Folder Structure

```
simple-MVP/
├── main.py                 # FastAPI server entry point
├── orchestrator.py         # LangGraph supervisor workflow
├── cli.py                  # CLI test script
├── test_flow.py            # Test without LLM
├── agents/
│   ├── scalp_agent.py      # ReAct scalp trading agent (Sonnet)
│   └── fallback_agent.py   # ReAct general queries agent (Haiku)
├── skills/
│   └── confluence.py       # V2.1 confluence calculator
├── mcp_servers/
│   └── market_data.py      # Finnhub market data tools
├── core/
│   └── tracing.py          # Phoenix tracing setup
├── static/
│   └── index.html          # Demo web UI
├── docs/
│   └── SCALING_GUIDE.md    # Guide for adding agents
├── requirements.txt        # Python dependencies
├── .env                    # Environment config
└── docker-compose.yml      # Phoenix container (optional)
```

---

## API Reference

### POST /analyze

Analyze a stock for scalp trading setup.

**Request:**
```json
{
  "query": "Analyze NVDA for scalp, RSI 35, volume 1.3x, at VWAP support"
}
```

**Response:**
```json
{
  "success": true,
  "query": "Analyze NVDA for scalp...",
  "response": "## NVDA Scalp Analysis...",
  "intent": "SCALP_ANALYSIS"
}
```

### GET /health

Health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "orchestrator": "ready",
    "scalp_agent": "ready",
    "tracing": "active"
  },
  "config": {
    "anthropic_key": "configured",
    "finnhub_key": "configured",
    "phoenix": "http://localhost:6006/v1/traces"
  }
}
```

### GET /docs

Interactive Swagger documentation.

---

## How It Works

### Request Flow Example

**User Query:** "Analyze NVDA for scalp, RSI 35, volume 1.3x, touching VWAP, uptrend"

**Step 1: Intent Classification (Haiku)**
```
Input: "Analyze NVDA for scalp..."
Output: SCALP_ANALYSIS
Latency: ~200ms
Cost: ~$0.001
```

**Step 2: Route to Scalp Agent**
```
Router: intent == SCALP_ANALYSIS → scalp_agent node
```

**Step 3: Scalp Agent ReAct Loop (Sonnet)**

```
Thought: I need to get the current NVDA price and SPY market context.

Action: get_stock_quote("NVDA")
Observation: {price: 187.68, high: 189.60, low: 186.82}

Action: get_spy_change()
Observation: {change_percent: 0.04, market_condition: "neutral"}

Thought: Now I have all the data. Let me calculate the confluence score.

Action: calculate_confluence(symbol="NVDA", price=187.68, ...)
Observation: {score: 7.5, verdict: "LONG_SCALP", confidence: "HIGH"}

Final Answer: ## NVDA Scalp Analysis - V2.1 Methodology...
```

**Step 4: Format Response**
```
Returns structured response with trade setup
```

**Total Latency:** ~3-5 seconds
**Total Cost:** ~$0.01-0.02

---

## V2.1 Methodology

### Confluence Scoring (10 points max)

| Factor | Points | Criteria |
|--------|--------|----------|
| **Volume** | 0-2 | ≥1.2x avg = 2pts, ≥0.75x = 1pt, <0.75x = NO_TRADE |
| **RSI** | 0-2 | ≤30 = 2pts (oversold), 31-40 = 1pt, ≥68 = WAIT |
| **VWAP** | 0-2 | Above + uptrend = 2pts, touching = 2pts, bounce = 1pt |
| **Support** | 0-2 | Within 1% = 2pts, within 2.5% = 1pt |
| **SPY** | 0-1 | >+0.5% = 1pt (risk-on), <-1% = 0 (risk-off) |
| **Catalyst** | 0-1 | Positive news/earnings = 1pt |

### Decision Rules

| Score | Verdict | Action |
|-------|---------|--------|
| ≥6 | LONG_SCALP (HIGH) | Enter position |
| ≥5 + catalyst | LONG_SCALP (MEDIUM) | Enter with caution |
| RSI ≥68 | WAIT | Overbought, wait for pullback |
| Volume <0.75x | NO_TRADE | Too thin, skip |
| <5 | NO_TRADE | Insufficient confluence |

### Trade Setup Calculations

```python
# Entry
entry = current_price

# Stop Loss (below low of day - buffer)
stop_loss = low - (0.3% * price)

# Targets (based on ATR or range)
range_size = high - low
target_1 = entry + (range_size * 1.0)  # 1R
target_2 = entry + (range_size * 1.75) # 1.75R

# Risk/Reward
risk = entry - stop_loss
reward = target_2 - entry
rr_ratio = reward / risk
```

### High-Beta Stock Rules

These stocks require **50% position size** and **tight stops**:
- TSLA, PLTR, SMCI, AMD, NVDA, COIN, MSTR, RIVN

---

## Extending the System

### Adding a New Agent

1. Create agent file in `agents/`:

```python
# agents/swing_agent.py
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

SWING_AGENT_PROMPT = """You are a swing trading expert..."""

def create_swing_agent():
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    tools = [get_stock_quote, calculate_swing_setup]
    return create_react_agent(model=model, tools=tools, prompt=SWING_AGENT_PROMPT)
```

2. Add node to orchestrator:

```python
# orchestrator.py
workflow.add_node("swing_agent", swing_agent_node)
workflow.add_conditional_edges(
    "classifier",
    route_to_agent,
    {
        "scalp_agent": "scalp_agent",
        "swing_agent": "swing_agent",  # New route
        "general_responder": "general_responder"
    }
)
```

3. Update classifier prompt to recognize new intent.

### Adding a New Tool

1. Create tool with `@tool` decorator:

```python
# skills/technical.py
from langchain_core.tools import tool

@tool
def calculate_fibonacci_levels(high: float, low: float) -> dict:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "0.236": high - (diff * 0.236),
        "0.382": high - (diff * 0.382),
        "0.500": high - (diff * 0.500),
        "0.618": high - (diff * 0.618),
    }
```

2. Add to agent's tool list:

```python
tools = [get_stock_quote, get_spy_change, calculate_confluence, calculate_fibonacci_levels]
```

### Adding a New Data Source

1. Create new tool in `mcp_servers/`:

```python
# mcp_servers/news_data.py
import requests
from langchain_core.tools import tool

@tool
def get_stock_news(symbol: str) -> dict:
    """Get latest news for a stock."""
    # Use NewsAPI, Finnhub news, or another source
    ...
```

2. Add to relevant agent's tools.

---

## Troubleshooting

### Common Issues

**1. "Authentication error" from Anthropic**
```
Error: 401 authentication_error
```
- Check `ANTHROPIC_API_KEY` in `.env`
- Verify key has credits at console.anthropic.com

**2. "Connection refused" to Phoenix**
```
Error: Connection refused localhost:6006
```
- Start Phoenix: `python -m phoenix.server.main serve`
- Or use Docker: `docker-compose up -d phoenix`

**3. "Rate limit" from Finnhub**
```
Error: 429 Too Many Requests
```
- Finnhub free tier: 60 calls/minute
- Add delays between requests or upgrade plan

**4. "state_modifier" error**
```
TypeError: create_react_agent() got unexpected keyword argument 'state_modifier'
```
- LangGraph API changed: use `prompt` instead of `state_modifier`

**5. Port 8000 already in use**
```bash
lsof -ti:8000 | xargs kill -9
python main.py
```

### Checking Logs

```bash
# Server logs
tail -f /tmp/server.log

# Phoenix traces
open http://localhost:6006
```

### Testing Without LLM

Use `test_flow.py` to verify tools and tracing work:

```bash
python test_flow.py
```

This tests:
- Finnhub API connection
- Confluence calculation
- Phoenix trace export

---

## Cost Analysis

| Component | Per Call | Notes |
|-----------|----------|-------|
| Haiku (classifier) | ~$0.001 | Fast, cheap routing |
| Sonnet (agent) | ~$0.01 | Quality reasoning |
| Finnhub | Free | 60 calls/min limit |
| Phoenix | Free | Self-hosted |
| **Total per analysis** | **~$0.01-0.02** | |

**Monthly estimate (100 analyses/day):**
- 3,000 analyses × $0.015 = **~$45/month**

---

## Links

- **Demo UI:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Phoenix Traces:** http://localhost:6006
- **Finnhub:** https://finnhub.io
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **Phoenix Docs:** https://docs.arize.com/phoenix

---

## Summary

This MVP demonstrates a complete agentic AI system with:

- **Supervisor Pattern** - LangGraph orchestrator routes to specialists
- **ReAct Agent** - Reasoning + Acting loop with tool calls
- **Real Data** - Finnhub API for live market quotes
- **Domain Logic** - V2.1 confluence scoring methodology
- **Observability** - Phoenix traces every step
- **Simple UI** - Web interface for demos
- **Low Cost** - ~$0.01-0.02 per analysis

The architecture is extensible - add new agents, tools, or data sources following the patterns shown.
