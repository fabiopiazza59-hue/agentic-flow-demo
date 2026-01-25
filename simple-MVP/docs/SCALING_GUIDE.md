# Scaling Agentic Systems: Adding Agents & Growth Patterns

A practical guide for extending the Scalp Trading MVP with new agents and scaling to production.

---

## Table of Contents

1. [Adding a New Agent](#1-adding-a-new-agent)
2. [Adding New Tools](#2-adding-new-tools)
3. [Scaling Patterns](#3-scaling-patterns)
4. [Hierarchical Orchestration](#4-hierarchical-orchestration)
5. [Parallel Agent Execution](#5-parallel-agent-execution)
6. [Cost Optimization](#6-cost-optimization)
7. [Production Considerations](#7-production-considerations)
8. [Quick Reference](#8-quick-reference)

---

## 1. Adding a New Agent

Adding a new specialist agent requires **3 simple steps** and approximately **30-50 lines of code**.

### Step 1: Create the Agent File

Create a new file in `agents/` directory:

```python
# agents/swing_agent.py
"""
Swing Trading Agent - Multi-day position analysis
"""
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

# Import tools this agent will use
from skills.confluence import calculate_confluence
from mcp_servers.market_data import get_stock_quote, get_spy_change

load_dotenv()

# ============================================================
# SYSTEM PROMPT - Define the agent's expertise
# ============================================================
SWING_AGENT_PROMPT = """You are an expert swing trading analyst.

Your specialty is identifying multi-day trading opportunities (2-10 day holds).

## Your Methodology:
- Look for stocks breaking out of consolidation patterns
- Identify key support/resistance levels for entries
- Use RSI divergences for timing
- Consider sector rotation and market breadth

## Position Sizing:
- Standard positions: 5-10% of portfolio
- High conviction: Up to 15%
- Always define stop loss before entry

## Your Process:
1. Get current stock data with get_stock_quote
2. Assess market conditions with get_spy_change
3. Analyze the setup and provide recommendation

Always include: Entry zone, stop loss, target levels, expected hold time.
"""


# ============================================================
# AGENT CREATION
# ============================================================
def create_swing_agent():
    """
    Create ReAct agent for swing trading analysis.

    Uses Claude Sonnet for quality reasoning about
    multi-day trading setups.
    """
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
    )

    # Tools available to this agent
    tools = [
        get_stock_quote,
        get_spy_change,
        calculate_confluence,  # Can reuse existing tools
        # Add swing-specific tools here
    ]

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SWING_AGENT_PROMPT,
    )

    return agent


# Singleton pattern for efficiency
_swing_agent = None

def get_swing_agent():
    """Get or create the swing agent singleton."""
    global _swing_agent
    if _swing_agent is None:
        _swing_agent = create_swing_agent()
    return _swing_agent
```

### Step 2: Update the Orchestrator

Modify `orchestrator.py` to include the new agent:

```python
# orchestrator.py

# 1. Add import at top
from agents.swing_agent import get_swing_agent

# 2. Update the classifier prompt to recognize new intent
classification_prompt = f"""Classify this message into one of these intents:

- SCALP_ANALYSIS: User wants scalp trading analysis (minutes to hours)
- SWING_ANALYSIS: User wants swing trading analysis (days to weeks)
- GENERAL: General greeting or non-trading question
- UNKNOWN: Cannot determine

User message: "{user_message}"

Respond with ONLY the intent label."""

# 3. Update the router function
def route_to_agent(state: OrchestratorState) -> Literal["scalp_agent", "swing_agent", "fallback_agent"]:
    """Route to appropriate agent based on intent."""
    intent = state.get("intent", "UNKNOWN")

    if intent == "SCALP_ANALYSIS":
        return "scalp_agent"
    elif intent == "SWING_ANALYSIS":
        return "swing_agent"
    else:
        return "fallback_agent"  # Handles GENERAL and UNKNOWN

# 4. Create the agent node function
async def swing_agent_node(state: OrchestratorState) -> dict:
    """Execute the swing trading agent."""
    agent = get_swing_agent()
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

# 5. Add node and edges in create_orchestrator()
def create_orchestrator():
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("classifier", classify_intent)
    workflow.add_node("scalp_agent", scalp_agent_node)
    workflow.add_node("swing_agent", swing_agent_node)  # ← NEW
    workflow.add_node("fallback_agent", fallback_agent_node)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_to_agent,
        {
            "scalp_agent": "scalp_agent",
            "swing_agent": "swing_agent",  # ← NEW
            "fallback_agent": "fallback_agent"
        }
    )
    workflow.add_edge("scalp_agent", "format_response")
    workflow.add_edge("swing_agent", "format_response")  # ← NEW
    workflow.add_edge("fallback_agent", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()
```

### Step 3: Fallback Agent Pattern (Recommended)

Instead of a simple responder function, use a proper ReAct agent for the fallback.
This provides better responses and can use tools for calculations, terminology lookup, etc.

```python
# agents/fallback_agent.py
"""
Fallback Agent - Handles general queries with tools
"""
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def get_system_info() -> dict:
    """Get information about system capabilities."""
    return {"capabilities": [...], "supported_intents": [...]}

@tool
def get_trading_terminology(term: str) -> dict:
    """Explain trading terms like RSI, VWAP, etc."""
    glossary = {"rsi": {...}, "vwap": {...}}
    return glossary.get(term.lower(), {"error": "Term not found"})

FALLBACK_PROMPT = """You are a helpful assistant for the trading platform.
Answer general questions, explain terminology, and guide users."""

def create_fallback_agent():
    # Use Haiku for cost efficiency - simple queries don't need Sonnet
    model = ChatAnthropic(model="claude-3-5-haiku-latest", max_tokens=1024)
    tools = [get_system_info, get_trading_terminology]
    return create_react_agent(model=model, tools=tools, prompt=FALLBACK_PROMPT)
```

**Key insight:** Use Haiku for fallback/simple agents, Sonnet for complex analysis agents.
This optimizes cost while maintaining quality where it matters.

### Summary: Adding an Agent

| Step | File | Changes |
|------|------|---------|
| 1 | `agents/new_agent.py` | Create agent with prompt + tools |
| 2 | `orchestrator.py` | Add import, intent, router case, node, edges |
| 3 | `agents/fallback_agent.py` | Update fallback to mention new capability (optional) |

**Total: ~50 lines of code**

### Model Selection Guide

| Agent Type | Recommended Model | Reasoning |
|------------|-------------------|-----------|
| Complex analysis (scalp, swing) | Claude Sonnet | Requires deep reasoning |
| Simple queries (fallback, help) | Claude Haiku | Fast, cost-efficient |
| Classification/routing | Claude Haiku | Pattern matching only |

---

## 2. Adding New Tools

Tools are functions that agents can call. Adding a new tool is even simpler.

### Creating a Tool

```python
# skills/technical_analysis.py
"""
Technical Analysis Tools
"""
from langchain_core.tools import tool


@tool
def calculate_fibonacci_levels(high: float, low: float) -> dict:
    """
    Calculate Fibonacci retracement levels for a price range.

    Args:
        high: The high price of the range
        low: The low price of the range

    Returns:
        Dictionary with Fibonacci levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
    """
    diff = high - low

    return {
        "high": high,
        "low": low,
        "levels": {
            "0.236": round(high - (diff * 0.236), 2),
            "0.382": round(high - (diff * 0.382), 2),
            "0.500": round(high - (diff * 0.500), 2),
            "0.618": round(high - (diff * 0.618), 2),
            "0.786": round(high - (diff * 0.786), 2),
        }
    }


@tool
def calculate_risk_reward(
    entry: float,
    stop_loss: float,
    target: float
) -> dict:
    """
    Calculate risk/reward ratio for a trade.

    Args:
        entry: Entry price
        stop_loss: Stop loss price
        target: Target price

    Returns:
        Risk/reward analysis with position sizing suggestion
    """
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)

    if risk == 0:
        return {"error": "Risk cannot be zero"}

    rr_ratio = reward / risk

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target": target,
        "risk_amount": round(risk, 2),
        "reward_amount": round(reward, 2),
        "risk_reward_ratio": f"{rr_ratio:.1f}:1",
        "recommendation": "favorable" if rr_ratio >= 2 else "unfavorable"
    }
```

### Adding Tool to an Agent

```python
# In agents/scalp_agent.py or any agent

from skills.technical_analysis import calculate_fibonacci_levels, calculate_risk_reward

tools = [
    get_stock_quote,
    get_spy_change,
    calculate_confluence,
    calculate_fibonacci_levels,  # ← Add new tool
    calculate_risk_reward,       # ← Add new tool
]
```

### Tool Best Practices

| Practice | Why |
|----------|-----|
| Clear docstrings | LLM uses these to understand the tool |
| Type hints | Enables automatic validation |
| Return dictionaries | Structured data for LLM to parse |
| Handle errors gracefully | Return error info, don't raise exceptions |
| Keep tools focused | One tool = one purpose |

---

## 3. Scaling Patterns

As your system grows, choose the right pattern for your needs:

### Pattern 1: Linear Scaling (Current)

```
Orchestrator → Agent 1
             → Agent 2
             → Agent 3
             → ...
```

**Best for:** Up to 10-15 agents with distinct, non-overlapping domains.

**Pros:** Simple, easy to understand, low latency
**Cons:** Classifier becomes complex with many intents

### Pattern 2: Hierarchical Scaling

```
Main Orchestrator
├── Trading Orchestrator
│   ├── Scalp Agent
│   ├── Swing Agent
│   └── Options Agent
├── Research Orchestrator
│   ├── News Agent
│   ├── Fundamentals Agent
│   └── Technical Agent
└── Risk Orchestrator
    ├── Position Sizing Agent
    └── Portfolio Risk Agent
```

**Best for:** 15+ agents, multiple domains, enterprise systems.

**Pros:** Better organization, specialized routing, team ownership
**Cons:** More complex, additional latency from routing layers

### Pattern 3: Dynamic Routing

```python
def dynamic_router(state):
    """LLM decides which agent(s) to call."""
    router_prompt = f"""Given this query: "{state['query']}"

    Which agents should handle this? Choose from:
    - scalp_agent: Short-term trades
    - swing_agent: Multi-day trades
    - risk_agent: Risk assessment
    - research_agent: Market research

    Return a JSON list of agent names."""

    response = router_llm.invoke(router_prompt)
    return json.loads(response.content)
```

**Best for:** Complex queries requiring multiple agents.

**Pros:** Flexible, handles ambiguous queries
**Cons:** Higher cost (LLM routing), potential inconsistency

### Pattern 4: Parallel Execution

```python
async def parallel_analysis(state):
    """Run multiple agents in parallel."""
    tasks = [
        scalp_agent.ainvoke(state),
        swing_agent.ainvoke(state),
        risk_agent.ainvoke(state),
    ]
    results = await asyncio.gather(*tasks)
    return aggregate_results(results)
```

**Best for:** Comprehensive analysis requiring multiple perspectives.

**Pros:** Faster than sequential, richer output
**Cons:** Higher cost (multiple LLM calls), need aggregation logic

---

## 4. Hierarchical Orchestration

For larger systems, create sub-orchestrators:

### Structure

```
simple-MVP/
├── orchestrators/
│   ├── main_orchestrator.py      # Top-level router
│   ├── trading_orchestrator.py   # Trading sub-domain
│   └── research_orchestrator.py  # Research sub-domain
├── agents/
│   ├── trading/
│   │   ├── scalp_agent.py
│   │   └── swing_agent.py
│   └── research/
│       ├── news_agent.py
│       └── fundamentals_agent.py
```

### Main Orchestrator

```python
# orchestrators/main_orchestrator.py
"""
Main Orchestrator - Routes to domain-specific sub-orchestrators
"""
from langgraph.graph import StateGraph, START, END
from orchestrators.trading_orchestrator import get_trading_orchestrator
from orchestrators.research_orchestrator import get_research_orchestrator


class MainState(TypedDict):
    messages: Annotated[list, operator.add]
    domain: str  # "trading", "research", "general"
    result: dict | None


def classify_domain(state: MainState) -> dict:
    """Classify into high-level domain."""
    # Use Haiku for fast domain classification
    model = ChatAnthropic(model="claude-3-5-haiku-latest")

    prompt = """Classify into domain:
    - TRADING: Buy/sell analysis, entries, setups
    - RESEARCH: News, fundamentals, market analysis
    - GENERAL: Greetings, help, other

    Query: {query}

    Respond with domain only."""

    response = model.invoke(prompt.format(query=state["messages"][-1].content))
    return {"domain": response.content.strip().upper()}


def route_to_domain(state: MainState):
    domain = state.get("domain", "GENERAL")
    return {
        "TRADING": "trading_orchestrator",
        "RESEARCH": "research_orchestrator",
    }.get(domain, "general_responder")


async def trading_orchestrator_node(state: MainState) -> dict:
    """Delegate to trading sub-orchestrator."""
    trading_orch = get_trading_orchestrator()
    result = await trading_orch.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"], "result": result.get("analysis_result")}


def create_main_orchestrator():
    workflow = StateGraph(MainState)

    workflow.add_node("domain_classifier", classify_domain)
    workflow.add_node("trading_orchestrator", trading_orchestrator_node)
    workflow.add_node("research_orchestrator", research_orchestrator_node)
    workflow.add_node("general_responder", general_responder)

    workflow.add_edge(START, "domain_classifier")
    workflow.add_conditional_edges("domain_classifier", route_to_domain, {
        "trading_orchestrator": "trading_orchestrator",
        "research_orchestrator": "research_orchestrator",
        "general_responder": "general_responder",
    })
    workflow.add_edge("trading_orchestrator", END)
    workflow.add_edge("research_orchestrator", END)
    workflow.add_edge("general_responder", END)

    return workflow.compile()
```

### Trading Sub-Orchestrator

```python
# orchestrators/trading_orchestrator.py
"""
Trading Orchestrator - Routes between trading specialists
"""

def classify_trading_intent(state) -> dict:
    """Classify specific trading intent."""
    # Scalp vs Swing vs Options vs Crypto...
    ...

def create_trading_orchestrator():
    workflow = StateGraph(TradingState)

    workflow.add_node("classifier", classify_trading_intent)
    workflow.add_node("scalp_agent", scalp_agent_node)
    workflow.add_node("swing_agent", swing_agent_node)
    workflow.add_node("options_agent", options_agent_node)

    # ... routing logic

    return workflow.compile()
```

### Benefits of Hierarchical Design

| Benefit | Description |
|---------|-------------|
| **Separation of concerns** | Each orchestrator handles one domain |
| **Team ownership** | Different teams can own different sub-graphs |
| **Easier testing** | Test sub-orchestrators in isolation |
| **Better scaling** | Add domains without touching main flow |
| **Reduced classifier complexity** | 3-5 options per level vs 20+ flat |

---

## 5. Parallel Agent Execution

When you need multiple agents to analyze simultaneously:

### Implementation

```python
# orchestrator.py
import asyncio


async def parallel_analysis_node(state: OrchestratorState) -> dict:
    """
    Run multiple agents in parallel and aggregate results.

    Use case: "Give me a complete analysis of NVDA"
    - Scalp agent: Short-term setup
    - Swing agent: Multi-day outlook
    - Risk agent: Position sizing
    """
    # Get all agents
    scalp_agent = get_scalp_agent()
    swing_agent = get_swing_agent()
    risk_agent = get_risk_agent()

    # Run in parallel
    results = await asyncio.gather(
        scalp_agent.ainvoke({"messages": state["messages"]}),
        swing_agent.ainvoke({"messages": state["messages"]}),
        risk_agent.ainvoke({"messages": state["messages"]}),
        return_exceptions=True  # Don't fail if one agent errors
    )

    # Aggregate results
    combined_analysis = aggregate_analyses(results)

    return {
        "messages": [AIMessage(content=combined_analysis)],
        "analysis_result": {"response": combined_analysis}
    }


def aggregate_analyses(results: list) -> str:
    """Combine multiple agent responses into one."""
    sections = []

    agent_names = ["Scalp Analysis", "Swing Analysis", "Risk Assessment"]

    for name, result in zip(agent_names, results):
        if isinstance(result, Exception):
            sections.append(f"## {name}\n*Analysis unavailable*")
        else:
            # Extract last message content
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content:
                    sections.append(f"## {name}\n{msg.content}")
                    break

    return "\n\n---\n\n".join(sections)
```

### When to Use Parallel Execution

| Use Case | Pattern |
|----------|---------|
| "Complete analysis of X" | All relevant agents |
| "Compare scalp vs swing" | Specific subset |
| Risk assessment on every trade | Background agent |
| News + technicals + fundamentals | Research bundle |

### Cost Consideration

Parallel execution multiplies LLM costs:
- 1 agent: ~$0.01
- 3 agents parallel: ~$0.03
- 5 agents parallel: ~$0.05

Use parallel execution selectively for high-value queries.

---

## 6. Cost Optimization

### LLM Selection Strategy

| Task | Model | Cost | Reasoning |
|------|-------|------|-----------|
| Intent classification | Haiku | $0.001 | Fast, simple task |
| Simple routing | Haiku | $0.001 | Pattern matching |
| Agent reasoning | Sonnet | $0.01 | Quality analysis |
| Complex synthesis | Sonnet/Opus | $0.01-0.10 | Deep reasoning |

### Code Example: Model Tiering

```python
def get_model_for_task(task_type: str):
    """Select appropriate model based on task complexity."""
    models = {
        "classify": ChatAnthropic(model="claude-3-5-haiku-latest", max_tokens=50),
        "route": ChatAnthropic(model="claude-3-5-haiku-latest", max_tokens=100),
        "analyze": ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=2048),
        "synthesize": ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096),
    }
    return models.get(task_type, models["analyze"])
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_classification(query_hash: str) -> str:
    """Cache repeated classifications."""
    # Classification logic
    ...

def classify_with_cache(query: str) -> str:
    query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
    return cached_classification(query_hash)
```

### Cost Monitoring

```python
# Track costs per request
def track_cost(model: str, input_tokens: int, output_tokens: int):
    costs = {
        "claude-3-5-haiku-latest": {"input": 0.00025, "output": 0.00125},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    }
    model_cost = costs.get(model, costs["claude-sonnet-4-20250514"])
    total = (input_tokens * model_cost["input"] + output_tokens * model_cost["output"]) / 1000

    # Log or store for monitoring
    print(f"[Cost] {model}: ${total:.4f}")
    return total
```

---

## 7. Production Considerations

### State Persistence

For production, add persistent state:

```python
# Using Redis for conversation memory
from langchain_redis import RedisChatMessageHistory

def get_message_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379"
    )
```

### Error Handling

```python
async def safe_agent_invoke(agent, state, fallback_message: str):
    """Invoke agent with error handling."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": state["messages"]}),
            timeout=30.0  # 30 second timeout
        )
        return result
    except asyncio.TimeoutError:
        return {"messages": [AIMessage(content="Analysis timed out. Please try again.")]}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"messages": [AIMessage(content=fallback_message)]}
```

### Rate Limiting

```python
from asyncio import Semaphore

# Limit concurrent LLM calls
llm_semaphore = Semaphore(10)

async def rate_limited_invoke(agent, state):
    async with llm_semaphore:
        return await agent.ainvoke(state)
```

### Health Checks

```python
@app.get("/health/detailed")
async def detailed_health():
    """Check all system components."""
    checks = {
        "api": True,
        "anthropic": await check_anthropic(),
        "finnhub": await check_finnhub(),
        "phoenix": await check_phoenix(),
        "redis": await check_redis(),
    }

    all_healthy = all(checks.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": checks
    }
```

---

## 8. Quick Reference

### Adding Components Cheatsheet

| Component | Files to Modify | Lines of Code |
|-----------|-----------------|---------------|
| New Agent | `agents/new.py`, `orchestrator.py` | ~50 |
| New Tool | `skills/new.py`, agent's tool list | ~20 |
| New Intent | `orchestrator.py` (classifier + router) | ~10 |
| Sub-orchestrator | `orchestrators/sub.py`, main orchestrator | ~80 |

### File Templates

**New Agent Template:**
```python
# agents/{name}_agent.py
PROMPT = """..."""

def create_{name}_agent():
    model = ChatAnthropic(model="claude-sonnet-4-20250514")
    tools = [...]
    return create_react_agent(model=model, tools=tools, prompt=PROMPT)

_agent = None
def get_{name}_agent():
    global _agent
    if _agent is None: _agent = create_{name}_agent()
    return _agent
```

**New Tool Template:**
```python
# skills/{name}.py
from langchain_core.tools import tool

@tool
def my_tool(param1: str, param2: float = 1.0) -> dict:
    """Tool description for LLM."""
    # Implementation
    return {"result": ...}
```

### Scaling Decision Tree

```
How many agents do you need?

├── 1-5 agents
│   └── Use current flat orchestrator
│
├── 6-15 agents
│   └── Consider grouping by domain
│   └── Use hierarchical if clear domains exist
│
├── 15+ agents
│   └── Hierarchical orchestration required
│   └── Consider sub-teams owning sub-orchestrators
│
└── Dynamic/unpredictable needs
    └── LLM-based dynamic routing
```

---

## Summary

| Aspect | Difficulty | Key Files |
|--------|------------|-----------|
| Add agent | Easy | `agents/`, `orchestrator.py` |
| Add tool | Very Easy | `skills/`, agent file |
| Hierarchical scaling | Medium | `orchestrators/` |
| Parallel execution | Medium | `orchestrator.py` |
| Production hardening | Medium-Hard | Multiple files |

The architecture is designed to scale. Start simple, add agents as needed, and evolve to hierarchical patterns when complexity demands it.
