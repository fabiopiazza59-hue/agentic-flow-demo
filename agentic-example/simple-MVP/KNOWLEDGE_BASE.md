# Agentic Systems Knowledge Base

Complete technical reference for building agentic AI systems using LangGraph, with annotated code examples from the Scalp Trading MVP.

---

## Part 1: Core Concepts

### What is an Agentic System?

An agentic AI system is one where an LLM:
1. **Reasons** about tasks and decides what actions to take
2. **Acts** by calling tools/APIs
3. **Observes** the results
4. **Iterates** until the goal is achieved

Unlike simple prompt→response flows, agents have **autonomy** to make decisions.

### Key Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **ReAct** | Reason + Act loop | Single-agent with tools |
| **Supervisor** | Central router dispatches to specialists | Multi-agent orchestration |
| **Hierarchical** | Nested supervisors | Complex workflows |
| **Swarm** | Peer agents hand off tasks | Collaborative problem-solving |

This MVP uses **Supervisor + ReAct**: an orchestrator routes to a ReAct agent.

---

## Part 2: LangGraph Fundamentals

### StateGraph Basics

LangGraph uses a **directed graph** where:
- **Nodes** are functions that process state
- **Edges** define the flow between nodes
- **State** is a TypedDict passed through the graph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

# 1. Define state schema
class MyState(TypedDict):
    messages: Annotated[list, operator.add]  # Reducer: accumulates messages
    result: str | None

# 2. Create graph
workflow = StateGraph(MyState)

# 3. Add nodes (functions)
workflow.add_node("process", process_function)
workflow.add_node("respond", respond_function)

# 4. Add edges
workflow.add_edge(START, "process")
workflow.add_edge("process", "respond")
workflow.add_edge("respond", END)

# 5. Compile and run
graph = workflow.compile()
result = graph.invoke({"messages": [], "result": None})
```

### State Reducers

The `Annotated[list, operator.add]` syntax defines a **reducer**:

```python
# Without reducer: state["messages"] is replaced each time
messages: list

# With reducer: new messages are appended
messages: Annotated[list, operator.add]
```

### Conditional Routing

Route to different nodes based on state:

```python
def router(state: MyState) -> str:
    if state["intent"] == "SCALP":
        return "scalp_agent"
    return "fallback_agent"

workflow.add_conditional_edges(
    "classifier",           # From node
    router,                 # Router function
    {                       # Mapping
        "scalp_agent": "scalp_agent",
        "fallback_agent": "fallback_agent"
    }
)
```

---

## Part 3: Orchestrator Implementation

### Full Code: `orchestrator.py`

```python
"""
Orchestrator Agent - LangGraph StateGraph implementing supervisor pattern
"""
import os
import operator
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from agents.scalp_agent import get_scalp_agent
from agents.fallback_agent import get_fallback_agent

load_dotenv()


# ============================================================
# STATE DEFINITION
# ============================================================
class OrchestratorState(TypedDict):
    """
    State passed through the orchestrator workflow.

    Attributes:
        messages: Conversation history (accumulates via operator.add)
        intent: Classified user intent (SCALP_ANALYSIS, GENERAL, UNKNOWN)
        analysis_result: Output from specialist agent
    """
    messages: Annotated[list, operator.add]
    intent: str
    analysis_result: dict | None


# ============================================================
# CLASSIFIER NODE
# ============================================================
def get_classifier_model():
    """Use Haiku for fast, cheap intent classification."""
    return ChatAnthropic(
        model="claude-3-5-haiku-latest",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=256,  # Short response needed
    )


def classify_intent(state: OrchestratorState) -> dict:
    """
    Classify user intent to route to appropriate agent.

    This is the first node in the graph. It:
    1. Extracts the user's message from state
    2. Asks Haiku to classify it
    3. Returns the intent to update state
    """
    model = get_classifier_model()

    # Extract last user message
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    # Classification prompt
    prompt = f"""Classify this message into one of these intents:

- SCALP_ANALYSIS: User wants stock scalp trading analysis
- GENERAL: General greeting or non-trading question
- UNKNOWN: Cannot determine

User: "{user_message}"

Respond with ONLY the intent label."""

    response = model.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().upper()

    # Normalize
    if "SCALP" in intent:
        intent = "SCALP_ANALYSIS"
    elif "GENERAL" in intent:
        intent = "GENERAL"
    else:
        intent = "UNKNOWN"

    print(f"[Orchestrator] Intent: {intent}")
    return {"intent": intent}


# ============================================================
# ROUTER FUNCTION
# ============================================================
def route_to_agent(state: OrchestratorState) -> Literal["scalp_agent", "fallback_agent"]:
    """
    Conditional routing based on classified intent.
    Returns the name of the next node to execute.
    """
    if state.get("intent") == "SCALP_ANALYSIS":
        return "scalp_agent"
    return "fallback_agent"


# ============================================================
# SPECIALIST NODES
# ============================================================
async def scalp_agent_node(state: OrchestratorState) -> dict:
    """
    Execute the scalp trading ReAct agent.

    This wraps the agent and extracts the response.
    """
    agent = get_scalp_agent()
    result = await agent.ainvoke({"messages": state["messages"]})

    # Extract response
    analysis_result = None
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            analysis_result = {"response": msg.content}
            break

    return {
        "messages": result.get("messages", []),
        "analysis_result": analysis_result
    }


async def fallback_agent_node(state: OrchestratorState) -> dict:
    """Execute the fallback agent for general queries."""
    agent = get_fallback_agent()
    result = await agent.ainvoke({"messages": state["messages"]})

    analysis_result = None
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            analysis_result = {"response": msg.content}
            break

    return {
        "messages": result.get("messages", []),
        "analysis_result": analysis_result
    }


def format_response(state: OrchestratorState) -> dict:
    """Final node - response already in messages."""
    return {}


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================
def create_orchestrator():
    """
    Build the LangGraph StateGraph.

    Flow:
    START → classifier → [scalp_agent | fallback_agent] → format_response → END
    """
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("classifier", classify_intent)
    workflow.add_node("scalp_agent", scalp_agent_node)
    workflow.add_node("fallback_agent", fallback_agent_node)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges("classifier", route_to_agent, {
        "scalp_agent": "scalp_agent",
        "fallback_agent": "fallback_agent"
    })
    workflow.add_edge("scalp_agent", "format_response")
    workflow.add_edge("fallback_agent", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()


# Singleton pattern
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_orchestrator()
    return _orchestrator
```

### Key Takeaways

1. **State accumulates** - Messages build up through the graph
2. **Classifier is fast** - Use Haiku for routing decisions
3. **Router is a function** - Returns string matching node names
4. **Async for agents** - ReAct agents should be awaited

---

## Part 4: ReAct Agent Implementation

### Full Code: `agents/scalp_agent.py`

```python
"""
Scalp Trading Agent - ReAct agent for V2.1 scalp analysis
"""
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from skills.confluence import calculate_confluence
from mcp_servers.market_data import get_stock_quote, get_spy_change

load_dotenv()

# ============================================================
# SYSTEM PROMPT
# ============================================================
SCALP_AGENT_PROMPT = """You are an expert scalp trading analyst using V2.1 methodology.

## V2.1 Confluence Scoring:
- Volume: >= 1.2x = 2pts, >= 0.75x = 1pt, < 0.75x = NO_TRADE
- RSI: <= 30 = 2pts (oversold), 31-40 = 1pt, >= 68 = WAIT
- VWAP: Above + uptrend = 2pts, at VWAP = 2pts, bounce = 1pt
- Support: Within 1% = 2pts, within 2.5% = 1pt
- SPY: > +0.5% = 1pt (risk-on)
- Catalyst: Positive news = 1pt

## Decision Rules:
- Score >= 6: LONG_SCALP (HIGH confidence)
- Score >= 5 + catalyst: LONG_SCALP (MEDIUM)
- RSI >= 68: WAIT (overbought)
- Volume < 0.75x: NO_TRADE

## High-Beta (50% size, tight stops):
TSLA, PLTR, SMCI, AMD, NVDA, COIN, MSTR, RIVN

## Your Process:
1. Get stock quote with get_stock_quote
2. Get SPY context with get_spy_change
3. Calculate confluence score
4. Return structured analysis

Always include: Ticker, price, score, verdict, trade setup if LONG_SCALP.
"""


# ============================================================
# AGENT CREATION
# ============================================================
def create_scalp_agent():
    """
    Create ReAct agent with trading tools.

    The agent will:
    1. Receive user query
    2. Decide which tools to call
    3. Process results
    4. Generate final response
    """
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",  # Better reasoning
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
    )

    tools = [
        get_stock_quote,      # Fetch real-time price
        get_spy_change,       # Market context
        calculate_confluence, # V2.1 scoring
    ]

    # create_react_agent handles the ReAct loop automatically
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SCALP_AGENT_PROMPT,  # System instructions
    )

    return agent


# Singleton
_scalp_agent = None

def get_scalp_agent():
    global _scalp_agent
    if _scalp_agent is None:
        _scalp_agent = create_scalp_agent()
    return _scalp_agent
```

### How ReAct Works

```
┌─────────────────────────────────────────────────────────────┐
│                      ReAct Loop                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User: "Analyze NVDA, RSI 35"                               │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ THOUGHT: I need NVDA price and SPY context  │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ ACTION: get_stock_quote("NVDA")             │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ OBSERVATION: {price: 187.68, high: 189}     │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ THOUGHT: Now I need SPY data                │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ ACTION: get_spy_change()                    │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ OBSERVATION: {change: 0.04%}                │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ THOUGHT: Ready to calculate confluence      │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ ACTION: calculate_confluence(...)           │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ OBSERVATION: {score: 7.5, LONG_SCALP}       │            │
│  └─────────────────────────────────────────────┘            │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────────────────────────────────┐            │
│  │ FINAL ANSWER: Structured trade setup        │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 5: Tools Implementation

### LangChain Tool Pattern

```python
from langchain_core.tools import tool

@tool
def my_tool(param1: str, param2: int = 10) -> dict:
    """
    Tool description shown to the LLM.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Dictionary with results
    """
    # Implementation
    return {"result": "value"}
```

The `@tool` decorator:
1. Parses the docstring for the LLM
2. Validates input parameters
3. Handles errors gracefully

### Full Code: `mcp_servers/market_data.py`

```python
"""
Market Data Tools - Finnhub API integration
"""
import os
import requests
from datetime import datetime
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


@tool
def get_stock_quote(symbol: str) -> dict:
    """
    Get current stock quote including price, high, low, and change.

    Args:
        symbol: Stock ticker symbol (e.g., "NVDA", "TSLA")

    Returns:
        Dictionary with price data and market status
    """
    symbol = symbol.upper().strip()

    if not FINNHUB_API_KEY:
        # Fallback to mock data
        return _mock_quote(symbol)

    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/quote",
            params={"symbol": symbol, "token": FINNHUB_API_KEY},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        # Finnhub returns: c=current, h=high, l=low, o=open, pc=prev close
        if data.get("c", 0) == 0:
            return {"error": f"No data for {symbol}"}

        change_pct = ((data["c"] - data["pc"]) / data["pc"]) * 100

        return {
            "symbol": symbol,
            "price": round(data["c"], 2),
            "high": round(data["h"], 2),
            "low": round(data["l"], 2),
            "open": round(data["o"], 2),
            "previous_close": round(data["pc"], 2),
            "change_percent": round(change_pct, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}


@tool
def get_spy_change() -> dict:
    """
    Get SPY ETF percentage change for market context.

    Returns:
        Dictionary with SPY change and market condition assessment
    """
    result = get_stock_quote.invoke("SPY")

    if "error" in result:
        return result

    change = result.get("change_percent", 0)

    # Assess market condition
    if change > 0.5:
        condition = "risk-on"
    elif change < -1.0:
        condition = "risk-off"
    else:
        condition = "neutral"

    return {
        "symbol": "SPY",
        "change_percent": change,
        "price": result.get("price"),
        "market_condition": condition,
        "timestamp": result.get("timestamp"),
    }


def _mock_quote(symbol: str) -> dict:
    """Fallback mock data when API unavailable."""
    mock_prices = {
        "NVDA": 187.50, "TSLA": 248.00, "AMD": 145.00,
        "SPY": 590.00, "AAPL": 195.00,
    }
    price = mock_prices.get(symbol, 100.00)
    return {
        "symbol": symbol,
        "price": price,
        "high": price * 1.02,
        "low": price * 0.98,
        "change_percent": 0.5,
        "source": "mock",
    }
```

### Full Code: `skills/confluence.py`

```python
"""
V2.1 Confluence Calculator - Core trading logic
"""
from langchain_core.tools import tool

# High-beta stocks requiring special handling
HIGH_BETA_STOCKS = {"TSLA", "PLTR", "SMCI", "AMD", "NVDA", "COIN", "MSTR", "RIVN"}


@tool
def calculate_confluence(
    symbol: str,
    price: float,
    high: float,
    low: float,
    rsi: int,
    volume_ratio: float,
    vwap_position: str,
    trend: str,
    support: float,
    spy_change: float,
    has_catalyst: bool = False
) -> dict:
    """
    Calculate V2.1 confluence score for scalp trading decision.

    Args:
        symbol: Stock ticker
        price: Current price
        high: Day's high
        low: Day's low
        rsi: RSI value (0-100)
        volume_ratio: Current vs average volume (1.0 = average)
        vwap_position: "above", "below", or "touching"
        trend: "up", "down", or "sideways"
        support: Nearest support level
        spy_change: SPY % change
        has_catalyst: Whether positive catalyst exists

    Returns:
        Complete analysis with score, verdict, and trade setup
    """
    score = 0.0
    factors = {}
    reasons = []

    # =============================================
    # FACTOR 1: VOLUME (0-2 points)
    # =============================================
    if volume_ratio < 0.75:
        return {
            "symbol": symbol,
            "score": 0,
            "verdict": "NO_TRADE",
            "confidence": "N/A",
            "reason": "Volume too thin (<0.75x average) - NO_TRADE signal"
        }
    elif volume_ratio >= 1.2:
        factors["volume"] = {"points": 2, "ratio": volume_ratio}
        score += 2
        reasons.append(f"Strong volume ({volume_ratio}x)")
    else:
        factors["volume"] = {"points": 1, "ratio": volume_ratio}
        score += 1
        reasons.append(f"Moderate volume ({volume_ratio}x)")

    # =============================================
    # FACTOR 2: RSI (0-2 points)
    # =============================================
    if rsi >= 68:
        return {
            "symbol": symbol,
            "score": score,
            "verdict": "WAIT",
            "confidence": "N/A",
            "reason": f"RSI overbought ({rsi}) - wait for pullback"
        }
    elif rsi <= 30:
        factors["rsi"] = {"points": 2, "value": rsi, "zone": "oversold"}
        score += 2
        reasons.append(f"RSI oversold ({rsi})")
    elif rsi <= 40:
        factors["rsi"] = {"points": 1, "value": rsi, "zone": "low"}
        score += 1
        reasons.append(f"RSI favorable ({rsi})")
    else:
        factors["rsi"] = {"points": 0, "value": rsi, "zone": "neutral"}
        reasons.append(f"RSI neutral ({rsi})")

    # =============================================
    # FACTOR 3: VWAP (0-2 points)
    # =============================================
    vwap_position = vwap_position.lower()
    if vwap_position == "touching":
        factors["vwap"] = {"points": 2, "position": "touching"}
        score += 2
        reasons.append("At VWAP - prime entry")
    elif vwap_position == "above" and trend == "up":
        factors["vwap"] = {"points": 2, "position": "above", "trend": "up"}
        score += 2
        reasons.append("Above VWAP in uptrend")
    elif vwap_position == "below" and trend == "up":
        factors["vwap"] = {"points": 1, "position": "below", "trend": "up"}
        score += 1
        reasons.append("VWAP bounce setup")
    else:
        factors["vwap"] = {"points": 0, "position": vwap_position, "trend": trend}

    # =============================================
    # FACTOR 4: SUPPORT PROXIMITY (0-2 points)
    # =============================================
    support_dist = abs(price - support) / price * 100
    if support_dist <= 1.0:
        factors["support"] = {"points": 2, "distance_pct": support_dist}
        score += 2
        reasons.append(f"Very close to support ({support_dist:.1f}%)")
    elif support_dist <= 2.5:
        factors["support"] = {"points": 1, "distance_pct": support_dist}
        score += 1
        reasons.append(f"Near support ({support_dist:.1f}%)")
    else:
        factors["support"] = {"points": 0, "distance_pct": support_dist}

    # =============================================
    # FACTOR 5: SPY CONTEXT (0-1 point)
    # =============================================
    if spy_change > 0.5:
        factors["spy"] = {"points": 1, "change": spy_change, "condition": "risk-on"}
        score += 1
        reasons.append(f"SPY bullish (+{spy_change}%)")
    elif spy_change < -1.0:
        factors["spy"] = {"points": 0, "change": spy_change, "condition": "risk-off"}
        reasons.append(f"SPY bearish ({spy_change}%)")
    else:
        factors["spy"] = {"points": 0.5, "change": spy_change, "condition": "neutral"}
        score += 0.5

    # =============================================
    # FACTOR 6: CATALYST (0-1 point)
    # =============================================
    if has_catalyst:
        factors["catalyst"] = {"points": 1, "present": True}
        score += 1
        reasons.append("Positive catalyst present")
    else:
        factors["catalyst"] = {"points": 0, "present": False}

    # =============================================
    # DETERMINE VERDICT
    # =============================================
    if score >= 6:
        verdict = "LONG_SCALP"
        confidence = "HIGH"
    elif score >= 5 and has_catalyst:
        verdict = "LONG_SCALP"
        confidence = "MEDIUM"
    else:
        verdict = "NO_TRADE"
        confidence = "LOW"

    # =============================================
    # CALCULATE TRADE SETUP (if LONG_SCALP)
    # =============================================
    trade_setup = None
    if verdict == "LONG_SCALP":
        range_size = high - low
        is_high_beta = symbol.upper() in HIGH_BETA_STOCKS

        trade_setup = {
            "entry": price,
            "stop_loss": round(low - (price * 0.003), 2),  # Below low + buffer
            "target_1": round(price + range_size, 2),       # 1R
            "target_2": round(price + (range_size * 1.75), 2),  # 1.75R
            "risk_reward": "3.0:1",
            "position_size": "50%" if is_high_beta else "100%",
            "is_high_beta": is_high_beta,
        }

    return {
        "symbol": symbol.upper(),
        "score": score,
        "max_score": 10,
        "verdict": verdict,
        "confidence": confidence,
        "factors": factors,
        "trade_setup": trade_setup,
        "reasoning": " | ".join(reasons),
    }
```

---

## Part 6: Phoenix Tracing

### Setup Code: `core/tracing.py`

```python
"""
Phoenix Tracing - OpenTelemetry integration
"""
import os
from dotenv import load_dotenv

load_dotenv()


def setup_tracing(project_name: str = "scalp-mvp"):
    """
    Initialize Phoenix tracing with auto-instrumentation.

    This captures:
    - LangGraph node transitions
    - LLM calls (tokens, latency)
    - Tool invocations
    - Agent reasoning steps
    """
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # Register with Phoenix collector
    tracer_provider = register(
        project_name=project_name,
        endpoint="http://localhost:6006/v1/traces",
    )

    # Auto-instrument LangChain/LangGraph
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[Tracing] Phoenix: http://localhost:6006")
    return tracer_provider


def get_tracer(name: str = "scalp-mvp"):
    """Get tracer for manual span creation."""
    from opentelemetry import trace
    return trace.get_tracer(name)
```

### What Phoenix Captures

```
┌─────────────────────────────────────────────────────────────┐
│ Trace: analyze_nvda_request                                 │
│ Duration: 4.2s                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ├── Span: orchestrator.invoke (4.2s)                       │
│ │   │                                                       │
│ │   ├── Span: classifier (0.3s)                            │
│ │   │   └── LLM: claude-3-5-haiku                          │
│ │   │       Input: 156 tokens                              │
│ │   │       Output: 2 tokens ("SCALP_ANALYSIS")            │
│ │   │                                                       │
│ │   ├── Span: scalp_agent (3.8s)                           │
│ │   │   │                                                   │
│ │   │   ├── Span: LLM call #1 (1.2s)                       │
│ │   │   │   └── claude-sonnet: "I need to get quote"       │
│ │   │   │                                                   │
│ │   │   ├── Span: tool.get_stock_quote (0.2s)              │
│ │   │   │   └── Input: {"symbol": "NVDA"}                  │
│ │   │   │   └── Output: {"price": 187.68, ...}             │
│ │   │   │                                                   │
│ │   │   ├── Span: tool.get_spy_change (0.1s)               │
│ │   │   │   └── Output: {"change": 0.04, ...}              │
│ │   │   │                                                   │
│ │   │   ├── Span: tool.calculate_confluence (0.01s)        │
│ │   │   │   └── Output: {"score": 7.5, ...}                │
│ │   │   │                                                   │
│ │   │   └── Span: LLM call #2 (2.1s)                       │
│ │   │       └── Final response generation                  │
│ │   │                                                       │
│ │   └── Span: format_response (0.001s)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Starting Phoenix

```bash
# Option 1: Python module (recommended)
python -m phoenix.server.main serve

# Option 2: Docker
docker run -p 6006:6006 arizephoenix/phoenix

# Option 3: Docker Compose
docker-compose up -d phoenix
```

---

## Part 7: FastAPI Integration

### Full Code: `main.py`

```python
"""
FastAPI Server - Entry point for the agentic system
"""
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

load_dotenv()

from core.tracing import setup_tracing
from orchestrator import get_orchestrator


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="Natural language query")


class AnalyzeResponse(BaseModel):
    success: bool
    query: str
    response: str
    intent: str | None = None


# ============================================================
# LIFESPAN (startup/shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tracing on startup."""
    print("=" * 50)
    print("  SCALP MVP - Agentic Trading Assistant")
    print("=" * 50)

    setup_tracing("scalp-mvp")

    print(f"[Server] API: http://localhost:{os.getenv('PORT', 8000)}")
    print(f"[Server] UI: http://localhost:{os.getenv('PORT', 8000)}")
    print(f"[Server] Docs: http://localhost:{os.getenv('PORT', 8000)}/docs")
    print("=" * 50)

    yield

    print("[Server] Shutting down...")


# ============================================================
# APP CREATION
# ============================================================
app = FastAPI(
    title="Scalp MVP",
    description="Agentic Scalp Trading Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================
@app.get("/")
async def root():
    """Serve the demo UI."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """Health check with config status."""
    return {
        "status": "healthy",
        "config": {
            "anthropic": "ok" if os.getenv("ANTHROPIC_API_KEY") else "missing",
            "finnhub": "ok" if os.getenv("FINNHUB_API_KEY") else "mock",
        }
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main endpoint - invokes the orchestrator.

    Flow:
    1. Receive query
    2. Invoke orchestrator graph
    3. Extract response from messages
    4. Return structured response
    """
    try:
        orchestrator = get_orchestrator()

        result = await orchestrator.ainvoke({
            "messages": [HumanMessage(content=request.query)],
            "intent": "",
            "analysis_result": None
        })

        # Extract response
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                response_text = msg.content
                break

        return AnalyzeResponse(
            success=True,
            query=request.query,
            response=response_text,
            intent=result.get("intent")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## Part 8: Quick Reference

### Starting the System

```bash
# Terminal 1: Phoenix
python -m phoenix.server.main serve

# Terminal 2: Server
cd simple-MVP
source venv/bin/activate
python main.py

# Browser
open http://localhost:8000      # Demo UI
open http://localhost:8000/docs # Swagger
open http://localhost:6006      # Phoenix
```

### CLI Testing

```bash
# Single query
python cli.py "Analyze NVDA for scalp, RSI 35"

# Full parameters
python cli.py "Check TSLA setup, RSI 42, volume 1.1x, touching VWAP, uptrend, support 245, earnings catalyst"
```

### Adding New Components

| Add | Steps |
|-----|-------|
| New Agent | 1. Create `agents/new_agent.py` 2. Add node to orchestrator 3. Update classifier |
| New Tool | 1. Create function with `@tool` 2. Add to agent's tool list |
| New Intent | 1. Add to classifier prompt 2. Add conditional edge in router |

### Common Imports

```python
# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# LangChain
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Tracing
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Agentic** | AI system that reasons and acts autonomously |
| **ReAct** | Reason + Act pattern for tool-using agents |
| **Supervisor** | Central agent that routes to specialists |
| **StateGraph** | LangGraph's directed graph with typed state |
| **Node** | Function in the graph that processes state |
| **Edge** | Connection between nodes defining flow |
| **Tool** | Function the LLM can call |
| **Span** | Unit of work in a trace (OpenTelemetry) |
| **Trace** | End-to-end request tracking |

---

*This knowledge base covers the complete implementation of an agentic trading system. Use it as a reference for building similar systems.*
