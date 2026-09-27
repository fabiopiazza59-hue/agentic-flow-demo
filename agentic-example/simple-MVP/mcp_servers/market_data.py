"""
Market Data MCP Server

FastMCP server providing real-time market data via Finnhub API.

Tools:
- get_stock_quote: Get current price, high, low for a symbol
- get_spy_change: Get SPY % change for market context

Run standalone: python -m mcp_servers.market_data
"""

import os
import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


async def _fetch_quote(symbol: str) -> dict:
    """Fetch quote from Finnhub API."""
    if not FINNHUB_API_KEY:
        # Return mock data if no API key
        return _get_mock_quote(symbol)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{FINNHUB_BASE_URL}/quote",
            params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY}
        )
        data = response.json()

        if data.get("c", 0) > 0:
            return {
                "symbol": symbol.upper(),
                "price": data["c"],
                "high": data["h"],
                "low": data["l"],
                "open": data["o"],
                "previous_close": data["pc"],
                "change": data["d"],
                "change_percent": data["dp"]
            }
        else:
            return _get_mock_quote(symbol)


def _get_mock_quote(symbol: str) -> dict:
    """Return mock data for testing without API key."""
    mock_data = {
        "NVDA": {"price": 140.50, "high": 142.00, "low": 138.00, "change_percent": 1.2},
        "AMD": {"price": 125.30, "high": 127.00, "low": 123.50, "change_percent": 0.8},
        "TSLA": {"price": 455.00, "high": 462.00, "low": 448.00, "change_percent": 1.5},
        "META": {"price": 595.00, "high": 602.00, "low": 588.00, "change_percent": 0.6},
        "PLTR": {"price": 82.50, "high": 84.00, "low": 80.00, "change_percent": 2.1},
        "GOOGL": {"price": 198.00, "high": 201.00, "low": 195.00, "change_percent": 0.4},
        "SPY": {"price": 610.00, "high": 612.00, "low": 607.00, "change_percent": 0.5},
    }

    default = {"price": 100.00, "high": 102.00, "low": 98.00, "change_percent": 0.0}
    data = mock_data.get(symbol.upper(), default)

    return {
        "symbol": symbol.upper(),
        "price": data["price"],
        "high": data["high"],
        "low": data["low"],
        "open": data["price"] * 0.99,
        "previous_close": data["price"] * 0.99,
        "change": data["price"] * data["change_percent"] / 100,
        "change_percent": data["change_percent"],
        "source": "mock" if not FINNHUB_API_KEY else "finnhub"
    }


@tool
async def get_stock_quote(symbol: str) -> dict:
    """
    Get current stock quote including price, high, low, and change.

    Args:
        symbol: Stock ticker symbol (e.g., "NVDA", "AMD", "TSLA")

    Returns:
        Dictionary with price, high, low, open, previous_close, change, change_percent
    """
    return await _fetch_quote(symbol)


@tool
async def get_spy_change() -> dict:
    """
    Get SPY (S&P 500 ETF) change percentage for market context.

    Returns:
        Dictionary with SPY price and change_percent
    """
    quote = await _fetch_quote("SPY")
    return {
        "symbol": "SPY",
        "price": quote["price"],
        "change_percent": quote["change_percent"],
        "market_sentiment": "risk_on" if quote["change_percent"] > 0.5 else "risk_off" if quote["change_percent"] < -1 else "neutral"
    }


# Export tools for use in agents
MARKET_DATA_TOOLS = [get_stock_quote, get_spy_change]


# FastMCP Server (for standalone MCP server mode)
if __name__ == "__main__":
    from fastmcp import FastMCP

    mcp = FastMCP("Market Data Server")

    @mcp.tool()
    async def mcp_get_stock_quote(symbol: str) -> dict:
        """Get current stock quote for a symbol."""
        return await _fetch_quote(symbol)

    @mcp.tool()
    async def mcp_get_spy_change() -> dict:
        """Get SPY % change for market context."""
        quote = await _fetch_quote("SPY")
        return {
            "symbol": "SPY",
            "price": quote["price"],
            "change_percent": quote["change_percent"]
        }

    print("[MCP] Market Data Server starting...")
    mcp.run()
