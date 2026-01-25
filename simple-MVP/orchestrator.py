"""
Orchestrator Agent

LangGraph StateGraph implementing the supervisor pattern:
1. Classifier: Determines user intent (SCALP_ANALYSIS or GENERAL)
2. Router: Routes to appropriate specialist agent
3. Agents: Scalp Agent (trading) or Fallback Agent (general queries)
4. Responder: Formats the final response

Flow: User Query -> Classifier -> [Scalp Agent | Fallback Agent] -> Responder -> Response
"""

import os
import operator
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from agents.scalp_agent import get_scalp_agent
from agents.fallback_agent import get_fallback_agent

load_dotenv()


# State definition
class OrchestratorState(TypedDict):
    """State passed through the orchestrator workflow."""
    messages: Annotated[list, operator.add]
    intent: str
    analysis_result: dict | None


# Intent classifier using Haiku (cost-efficient)
def get_classifier_model():
    return ChatAnthropic(
        model="claude-3-5-haiku-latest",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=256,
    )


def classify_intent(state: OrchestratorState) -> dict:
    """
    Classify the user's intent to route to the appropriate agent.

    Intents:
    - SCALP_ANALYSIS: User wants to analyze a stock for scalp trading
    - GENERAL: General question or greeting
    - UNKNOWN: Cannot determine intent
    """
    model = get_classifier_model()

    # Get the last user message
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    classification_prompt = f"""Classify the following user message into one of these intents:

- SCALP_ANALYSIS: User wants to analyze a stock for scalp trading, mentions stock symbols, RSI, volume, VWAP, support levels, or trading setup
- GENERAL: General greeting, help request, or non-trading question
- UNKNOWN: Cannot determine intent

User message: "{user_message}"

Respond with ONLY the intent label (SCALP_ANALYSIS, GENERAL, or UNKNOWN)."""

    response = model.invoke([HumanMessage(content=classification_prompt)])
    intent = response.content.strip().upper()

    # Normalize intent
    if "SCALP" in intent or "ANALYSIS" in intent:
        intent = "SCALP_ANALYSIS"
    elif "GENERAL" in intent:
        intent = "GENERAL"
    else:
        intent = "UNKNOWN"

    print(f"[Orchestrator] Intent classified: {intent}")

    return {"intent": intent}


def route_to_agent(state: OrchestratorState) -> Literal["scalp_agent", "fallback_agent"]:
    """Route to the appropriate agent based on intent."""
    intent = state.get("intent", "UNKNOWN")

    if intent == "SCALP_ANALYSIS":
        return "scalp_agent"
    else:
        return "fallback_agent"


async def scalp_agent_node(state: OrchestratorState) -> dict:
    """Execute the scalp trading agent."""
    agent = get_scalp_agent()

    # Invoke the agent with the current messages
    result = await agent.ainvoke({"messages": state["messages"]})

    # Extract the analysis result from agent messages
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
    """Execute the fallback agent for general/unknown intents."""
    agent = get_fallback_agent()

    # Invoke the agent with the current messages
    result = await agent.ainvoke({"messages": state["messages"]})

    # Extract the response from agent messages
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
    """Format the final response from the analysis."""
    # Response is already in the messages, just pass through
    return {}


def create_orchestrator():
    """
    Create the orchestrator workflow using LangGraph StateGraph.

    Flow:
    START -> classifier -> [scalp_agent | fallback_agent] -> format_response -> END
    """
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("classifier", classify_intent)
    workflow.add_node("scalp_agent", scalp_agent_node)
    workflow.add_node("fallback_agent", fallback_agent_node)
    workflow.add_node("format_response", format_response)

    # Add edges
    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_to_agent,
        {
            "scalp_agent": "scalp_agent",
            "fallback_agent": "fallback_agent"
        }
    )
    workflow.add_edge("scalp_agent", "format_response")
    workflow.add_edge("fallback_agent", "format_response")
    workflow.add_edge("format_response", END)

    # Compile the workflow
    return workflow.compile()


# Singleton orchestrator instance
_orchestrator = None


def get_orchestrator():
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = create_orchestrator()
    return _orchestrator
