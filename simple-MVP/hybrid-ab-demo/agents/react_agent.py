"""
ReAct Agent for Simple Queries

Tool-based reasoning agent for fast lookups and simple calculations.
Routes here when complexity_score <= 3.

Handles:
- Market data lookups (stock quotes)
- Simple calculations
- Definitions and explanations
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from mcp_servers.market_data import get_stock_quote, get_spy_change

load_dotenv()


# ============================================================
# ADDITIONAL TOOLS FOR REACT AGENT
# ============================================================

@tool
def calculate(expression: str) -> dict:
    """
    Perform financial calculations.

    Args:
        expression: Math expression (e.g., "100000 * 0.07" or "50000 / 12")

    Returns:
        Calculation result
    """
    try:
        # Safe evaluation of math expressions
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return {"error": "Only basic math operations allowed"}

        result = eval(expression)
        return {
            "expression": expression,
            "result": round(result, 2) if isinstance(result, float) else result
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_financial_term(term: str) -> dict:
    """
    Explain a financial term or concept.

    Args:
        term: The term to explain (e.g., "Sharpe ratio", "Monte Carlo")

    Returns:
        Definition and explanation
    """
    glossary = {
        "sharpe ratio": {
            "name": "Sharpe Ratio",
            "definition": "Risk-adjusted return measure = (Return - Risk-free rate) / Standard deviation",
            "interpretation": "Higher is better. Above 1 is good, above 2 is very good."
        },
        "monte carlo": {
            "name": "Monte Carlo Simulation",
            "definition": "Statistical technique using random sampling to model probability of outcomes",
            "use_case": "Retirement planning, portfolio risk analysis, option pricing"
        },
        "var": {
            "name": "Value at Risk (VaR)",
            "definition": "Maximum expected loss over a time period at a confidence level",
            "example": "95% VaR of $10,000 means 95% confident you won't lose more than $10K"
        },
        "portfolio optimization": {
            "name": "Portfolio Optimization",
            "definition": "Mathematical process to select optimal asset allocation",
            "methods": "Mean-variance (Markowitz), Black-Litterman, Risk parity"
        },
        "compound interest": {
            "name": "Compound Interest",
            "definition": "Interest calculated on initial principal plus accumulated interest",
            "formula": "A = P(1 + r/n)^(nt)"
        },
        "diversification": {
            "name": "Diversification",
            "definition": "Spreading investments across assets to reduce risk",
            "benefit": "Reduces portfolio volatility without necessarily reducing returns"
        },
        "etf": {
            "name": "Exchange-Traded Fund (ETF)",
            "definition": "Investment fund traded on stock exchanges like individual stocks",
            "examples": "SPY (S&P 500), QQQ (Nasdaq), VTI (Total Market)"
        },
        "expense ratio": {
            "name": "Expense Ratio",
            "definition": "Annual fee charged by funds as percentage of assets",
            "guidance": "Lower is better. Index funds often <0.1%, active funds 0.5-1.5%"
        }
    }

    term_lower = term.lower().strip()

    # Exact match
    if term_lower in glossary:
        return glossary[term_lower]

    # Partial match
    for key, value in glossary.items():
        if term_lower in key or key in term_lower:
            return value

    return {
        "term": term,
        "note": "Term not found. Try: Sharpe ratio, Monte Carlo, VaR, portfolio optimization, compound interest"
    }


# ============================================================
# SYSTEM PROMPT
# ============================================================

REACT_AGENT_PROMPT = """You are a helpful financial assistant that answers simple queries quickly.

You have access to tools for:
- Getting real-time stock quotes (get_stock_quote, get_spy_change)
- Performing calculations (calculate)
- Explaining financial terms (get_financial_term)

## Your Role
Answer straightforward financial questions using your tools. Be concise and direct.

## Guidelines
1. For stock prices, use get_stock_quote with the ticker symbol
2. For market context, use get_spy_change
3. For math, use the calculate tool
4. For definitions, use get_financial_term
5. Always cite the source of market data

## Response Style
- Keep responses brief and to the point
- Include relevant numbers and facts
- Don't over-explain simple lookups

## Examples
- "What's Apple's stock price?" → Use get_stock_quote("AAPL")
- "Calculate 10% of 50000" → Use calculate("50000 * 0.10")
- "What is a Sharpe ratio?" → Use get_financial_term("sharpe ratio")
"""


# ============================================================
# AGENT CREATION
# ============================================================

def create_react_agent_instance():
    """
    Create the ReAct agent for simple queries.

    Uses Claude Haiku for cost efficiency on simple lookups.
    """
    model = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )

    tools = [
        get_stock_quote,
        get_spy_change,
        calculate,
        get_financial_term,
    ]

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=REACT_AGENT_PROMPT,
    )

    return agent


# Singleton pattern
_react_agent = None


def get_react_agent():
    """Get or create the ReAct agent singleton."""
    global _react_agent
    if _react_agent is None:
        _react_agent = create_react_agent_instance()
    return _react_agent
