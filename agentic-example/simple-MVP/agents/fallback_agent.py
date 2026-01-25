"""
Fallback Agent

General-purpose ReAct agent that handles all non-trading questions.
Uses web search and general knowledge to answer user queries.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

load_dotenv()


# ============================================================
# TOOLS FOR GENERAL QUERIES
# ============================================================
@tool
def get_system_info() -> dict:
    """
    Get information about the trading assistant capabilities.

    Returns:
        Dictionary with system capabilities and available features
    """
    return {
        "name": "Scalp Trading Assistant",
        "version": "1.0.0",
        "capabilities": [
            "Scalp trading analysis using V2.1 methodology",
            "Real-time stock quotes via Finnhub",
            "SPY market context analysis",
            "Confluence scoring (6 factors)",
            "Trade setup with entry, stop, targets"
        ],
        "supported_intents": [
            "SCALP_ANALYSIS - Short-term trading setups",
            "GENERAL - General questions (you are here)"
        ],
        "high_beta_stocks": ["TSLA", "PLTR", "SMCI", "AMD", "NVDA", "COIN", "MSTR", "RIVN"],
        "methodology": "V2.1 Confluence Scoring"
    }


@tool
def calculate_basic_math(expression: str) -> dict:
    """
    Perform basic mathematical calculations.

    Args:
        expression: A math expression like "100 * 1.5" or "500 / 4"

    Returns:
        The calculation result
    """
    try:
        # Safe evaluation of basic math
        allowed_chars = set("0123456789+-*/.(). ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Only basic math operations allowed (+, -, *, /)"}

        result = eval(expression)
        return {
            "expression": expression,
            "result": round(result, 4) if isinstance(result, float) else result
        }
    except Exception as e:
        return {"error": f"Could not calculate: {str(e)}"}


@tool
def get_trading_terminology(term: str) -> dict:
    """
    Explain common trading terms and concepts.

    Args:
        term: The trading term to explain (e.g., "RSI", "VWAP", "scalping")

    Returns:
        Definition and explanation of the term
    """
    glossary = {
        "rsi": {
            "name": "Relative Strength Index (RSI)",
            "definition": "A momentum oscillator measuring speed and magnitude of price changes",
            "range": "0-100",
            "interpretation": "Below 30 = oversold, Above 70 = overbought"
        },
        "vwap": {
            "name": "Volume Weighted Average Price",
            "definition": "Average price weighted by volume, used as a trading benchmark",
            "use": "Institutional traders use VWAP to gauge trade execution quality"
        },
        "scalping": {
            "name": "Scalp Trading",
            "definition": "A trading style aiming for small profits on short-term trades",
            "timeframe": "Seconds to minutes",
            "key": "High win rate with small gains per trade"
        },
        "confluence": {
            "name": "Confluence",
            "definition": "Multiple technical factors aligning to support a trade decision",
            "example": "RSI oversold + price at support + high volume = high confluence"
        },
        "support": {
            "name": "Support Level",
            "definition": "A price level where buying pressure prevents further decline",
            "trading": "Often used as entry points for long positions"
        },
        "resistance": {
            "name": "Resistance Level",
            "definition": "A price level where selling pressure prevents further rise",
            "trading": "Often used as profit targets or short entry points"
        },
        "stop loss": {
            "name": "Stop Loss",
            "definition": "A predetermined price level to exit a losing trade",
            "purpose": "Risk management - limits potential loss on a trade"
        },
        "high beta": {
            "name": "High Beta Stock",
            "definition": "A stock with higher volatility than the overall market",
            "trading": "Requires smaller position sizes and tighter stops"
        }
    }

    term_lower = term.lower().strip()

    # Check for exact match or partial match
    if term_lower in glossary:
        return glossary[term_lower]

    # Try partial matching
    for key, value in glossary.items():
        if term_lower in key or key in term_lower:
            return value

    return {
        "term": term,
        "error": "Term not found in glossary",
        "available_terms": list(glossary.keys())
    }


# ============================================================
# SYSTEM PROMPT
# ============================================================
FALLBACK_AGENT_PROMPT = """You are a helpful assistant for the Scalp Trading platform.

Your role is to answer general questions that are NOT about specific stock analysis.
You help users understand:
- How the trading assistant works
- Trading terminology and concepts
- Basic calculations for position sizing
- General questions about the platform

## Your Tools:
- get_system_info: Get details about platform capabilities
- calculate_basic_math: Perform calculations
- get_trading_terminology: Explain trading terms

## Guidelines:
1. Be helpful and conversational
2. If asked about specific stock analysis, suggest using a scalp analysis query
3. For trading terms, use the glossary tool
4. Keep responses concise but informative

## Example Responses:
- "What is RSI?" → Use get_trading_terminology
- "What can you do?" → Use get_system_info
- "Calculate 1000 * 0.02" → Use calculate_basic_math
- "Analyze NVDA" → Redirect to scalp analysis

Always be friendly and guide users to the right feature for their needs.
"""


# ============================================================
# AGENT CREATION
# ============================================================
def create_fallback_agent():
    """
    Create the fallback ReAct agent for general queries.

    Uses Claude Haiku for cost efficiency since these are
    simpler queries that don't require deep analysis.
    """
    model = ChatAnthropic(
        model="claude-3-5-haiku-latest",  # Cost efficient for simple queries
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )

    tools = [
        get_system_info,
        calculate_basic_math,
        get_trading_terminology,
    ]

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=FALLBACK_AGENT_PROMPT,
    )

    return agent


# Singleton pattern
_fallback_agent = None


def get_fallback_agent():
    """Get or create the fallback agent singleton."""
    global _fallback_agent
    if _fallback_agent is None:
        _fallback_agent = create_fallback_agent()
    return _fallback_agent
