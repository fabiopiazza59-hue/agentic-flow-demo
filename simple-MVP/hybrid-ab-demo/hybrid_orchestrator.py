"""
Hybrid Orchestrator - ReAct + CodeAct Routing

LangGraph StateGraph implementing hybrid architecture:
1. Classifier: Determines intent and complexity
2. Router: Routes to ReAct (simple) or CodeAct (complex)
3. Agents: Execute and return results
4. Responder: Format final response with path info

Flow: Query -> Classify -> [ReAct | CodeAct] -> Response
"""

import os
import operator
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from agents.react_agent import get_react_agent
from agents.codeact_agent import get_codeact_agent

load_dotenv()


# ============================================================
# STATE DEFINITION
# ============================================================

class HybridState(TypedDict):
    """State passed through the hybrid orchestrator."""
    messages: Annotated[list, operator.add]
    query: str
    intent: str
    complexity_score: int
    path: str  # "react" or "codeact"
    result: dict | None


# ============================================================
# COMPLEXITY ROUTING RULES
# ============================================================

# Patterns that indicate simple queries (ReAct path)
REACT_PATTERNS = [
    "what is",
    "what's",
    "price of",
    "stock price",
    "quote for",
    "how much",
    "explain",
    "define",
    "calculate",
    "what does",
    "meaning of",
]

# Patterns that indicate complex queries (CodeAct path)
CODEACT_PATTERNS = [
    "monte carlo",
    "simulation",
    "simulate",
    "optimize",
    "optimization",
    "backtest",
    "analyze trend",
    "statistical analysis",
    "probability",
    "projection",
    "forecast",
    "compare performance",
    "risk analysis",
    "portfolio optimization",
    "sharpe ratio",
    "what if",
    "scenario",
]


# ============================================================
# CLASSIFIER NODE
# ============================================================

def get_classifier_model():
    """Fast model for classification."""
    return ChatAnthropic(
        model="claude-3-5-haiku-latest",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=256,
    )


def classify_and_route(state: HybridState) -> dict:
    """
    Classify the query's intent and complexity.

    Complexity Score:
    - 1-3: Simple (ReAct) - lookups, definitions, basic calculations
    - 4-7: Complex (CodeAct) - simulations, optimization, analysis

    Returns updated state with intent, complexity_score, and path.
    """
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content.lower()
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            query = msg.get("content", "").lower()
            break

    # Quick pattern matching first (no LLM needed for obvious cases)
    for pattern in CODEACT_PATTERNS:
        if pattern in query:
            print(f"[Orchestrator] Pattern match: '{pattern}' -> CodeAct path")
            return {
                "query": query,
                "intent": "COMPLEX_ANALYSIS",
                "complexity_score": 5,
                "path": "codeact"
            }

    for pattern in REACT_PATTERNS:
        if pattern in query:
            print(f"[Orchestrator] Pattern match: '{pattern}' -> ReAct path")
            return {
                "query": query,
                "intent": "SIMPLE_QUERY",
                "complexity_score": 2,
                "path": "react"
            }

    # Use LLM for ambiguous cases
    model = get_classifier_model()

    classification_prompt = f"""Analyze this financial query and determine its complexity.

Query: "{query}"

Rate the complexity from 1-7:
- 1-3 (SIMPLE): Stock price lookups, definitions, basic calculations, explanations
- 4-7 (COMPLEX): Simulations, optimization, statistical analysis, projections, multi-step computations

Respond in this exact format:
INTENT: <brief intent description>
COMPLEXITY: <number 1-7>
REASONING: <one sentence explanation>"""

    response = model.invoke([HumanMessage(content=classification_prompt)])
    content = response.content.strip()

    # Parse response
    lines = content.split("\n")
    intent = "UNKNOWN"
    complexity = 3  # Default to borderline

    for line in lines:
        if line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip()
        elif line.startswith("COMPLEXITY:"):
            try:
                complexity = int(line.replace("COMPLEXITY:", "").strip())
            except ValueError:
                complexity = 3

    # Determine path
    path = "codeact" if complexity >= 4 else "react"

    print(f"[Orchestrator] LLM Classification:")
    print(f"  Intent: {intent}")
    print(f"  Complexity: {complexity}/7")
    print(f"  Path: {path.upper()}")

    return {
        "query": query,
        "intent": intent,
        "complexity_score": complexity,
        "path": path
    }


def route_to_path(state: HybridState) -> Literal["react_node", "codeact_node"]:
    """Route to the appropriate agent based on path."""
    if state.get("path") == "codeact":
        return "codeact_node"
    return "react_node"


# ============================================================
# AGENT NODES
# ============================================================

async def react_node(state: HybridState) -> dict:
    """Execute the ReAct agent for simple queries."""
    print("[Orchestrator] Executing ReAct agent...")

    agent = get_react_agent()

    # Invoke the agent with messages
    result = await agent.ainvoke({"messages": state["messages"]})

    # Extract response
    response_text = ""
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.content:
            response_text = msg.content
            break

    return {
        "messages": result.get("messages", []),
        "result": {
            "success": True,
            "response": response_text,
            "path": "react"
        }
    }


async def codeact_node(state: HybridState) -> dict:
    """Execute the CodeAct agent for complex analysis."""
    print("[Orchestrator] Executing CodeAct agent...")

    agent = get_codeact_agent()

    # Get the original query
    query = state.get("query", "")
    if not query:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

    # Run CodeAct agent
    result = await agent.run(query)

    # Format response for display
    if result.get("success"):
        response_parts = []

        if result.get("explanation"):
            response_parts.append(f"**Approach:** {result['explanation']}")

        if result.get("output"):
            response_parts.append(f"\n**Results:**\n```\n{result['output']}```")

        if result.get("result"):
            response_parts.append(f"\n**Data:** {result['result']}")

        response_text = "\n".join(response_parts)
    else:
        response_text = f"Analysis failed: {result.get('error', 'Unknown error')}"

    return {
        "messages": [AIMessage(content=response_text)],
        "result": result
    }


# ============================================================
# RESPONSE FORMATTER
# ============================================================

def format_response(state: HybridState) -> dict:
    """Format the final response with path information."""
    # Response is already in messages
    return {}


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

def create_hybrid_orchestrator():
    """
    Create the hybrid orchestrator workflow.

    Flow:
    START -> classifier -> [react_node | codeact_node] -> format_response -> END
    """
    workflow = StateGraph(HybridState)

    # Add nodes
    workflow.add_node("classifier", classify_and_route)
    workflow.add_node("react_node", react_node)
    workflow.add_node("codeact_node", codeact_node)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_to_path,
        {
            "react_node": "react_node",
            "codeact_node": "codeact_node"
        }
    )
    workflow.add_edge("react_node", "format_response")
    workflow.add_edge("codeact_node", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()


# Singleton
_orchestrator = None


def get_hybrid_orchestrator():
    """Get or create the hybrid orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_hybrid_orchestrator()
    return _orchestrator


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        orchestrator = get_hybrid_orchestrator()

        # Test simple query (ReAct)
        print("\n" + "=" * 60)
        print("TEST 1: Simple Query (should use ReAct)")
        print("=" * 60)

        result1 = await orchestrator.ainvoke({
            "messages": [HumanMessage(content="What's Apple's stock price?")],
            "query": "",
            "intent": "",
            "complexity_score": 0,
            "path": "",
            "result": None
        })

        print(f"\nPath used: {result1.get('path', 'unknown')}")
        print(f"Intent: {result1.get('intent', 'unknown')}")

        # Test complex query (CodeAct)
        print("\n" + "=" * 60)
        print("TEST 2: Complex Query (should use CodeAct)")
        print("=" * 60)

        result2 = await orchestrator.ainvoke({
            "messages": [HumanMessage(content="Run a Monte Carlo simulation for retirement with $100k initial investment over 20 years")],
            "query": "",
            "intent": "",
            "complexity_score": 0,
            "path": "",
            "result": None
        })

        print(f"\nPath used: {result2.get('path', 'unknown')}")
        print(f"Intent: {result2.get('intent', 'unknown')}")

        # Show response excerpt
        for msg in result2.get("messages", [])[-1:]:
            if hasattr(msg, "content"):
                print(f"\nResponse excerpt:\n{msg.content[:500]}...")

    asyncio.run(test())
