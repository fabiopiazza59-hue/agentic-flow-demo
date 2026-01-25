"""
Scalp Trading Agent

ReAct agent specialized in scalp trading analysis using V2.1 methodology.
Uses tools to fetch market data and calculate confluence scores.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from skills.confluence import calculate_confluence
from mcp_servers.market_data import get_stock_quote, get_spy_change

load_dotenv()

# System prompt for the scalp trading agent
SCALP_AGENT_PROMPT = """You are an expert scalp trading analyst using the V2.1 methodology.

Your job is to analyze stocks for potential scalp trading setups using a 6-factor confluence scoring system.

## V2.1 Methodology Overview:
- Volume: >= 1.2x = 2pts, >= 0.75x = 1pt, < 0.75x = NO_TRADE
- RSI: <= 30 = 2pts (oversold), 31-40 = 1pt, >= 68 = WAIT (overbought)
- VWAP: Above + uptrend = 2pts, at VWAP = 2pts, bounce setup = 1pt
- Support: Within 1% = 2pts, within 2.5% = 1pt
- SPY: > +0.5% = 1pt (risk-on), < -1% = 0 (risk-off)
- Catalyst: Positive news = 1pt

## Decision Rules:
- Score >= 6: LONG_SCALP (HIGH confidence)
- Score >= 5 + catalyst: LONG_SCALP (MEDIUM confidence)
- RSI >= 68: WAIT (overbought)
- Volume < 0.75x: NO_TRADE (too thin)

## High-Beta Stocks (50% position, tight stops):
TSLA, PLTR, SMCI, AMD, NVDA, COIN, MSTR, RIVN

## Your Process:
1. First, get the stock quote using get_stock_quote
2. Get SPY change using get_spy_change for market context
3. Use calculate_confluence with all the data to get the score and verdict
4. Return a clear analysis with the trade setup if applicable

Always provide a structured response with:
- Ticker and current price
- Confluence score and verdict
- Trade setup (entry, stop, targets) if LONG_SCALP
- Brief reasoning for the decision
"""


def create_scalp_agent():
    """
    Create the scalp trading ReAct agent with market data and confluence tools.

    Returns:
        Compiled ReAct agent ready to invoke
    """
    # Use Claude Sonnet for better reasoning
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2048,
    )

    # Tools available to the agent
    tools = [
        get_stock_quote,
        get_spy_change,
        calculate_confluence,
    ]

    # Create ReAct agent with system prompt
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SCALP_AGENT_PROMPT,
    )

    return agent


# Singleton agent instance
_scalp_agent = None


def get_scalp_agent():
    """Get or create the scalp agent singleton."""
    global _scalp_agent
    if _scalp_agent is None:
        _scalp_agent = create_scalp_agent()
    return _scalp_agent
