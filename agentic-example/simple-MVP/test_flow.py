#!/usr/bin/env python3
"""
Test Flow - Demonstrates the agentic workflow without requiring LLM credits.

This script tests:
1. Phoenix tracing (spans are logged)
2. Market Data MCP tools (Finnhub API)
3. Confluence Skill (V2.1 calculator)
4. Full data flow

Run: python test_flow.py
View traces: http://localhost:6006
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Phoenix tracing FIRST
from core.tracing import setup_tracing
tracer_provider = setup_tracing("scalp-mvp-test")

from opentelemetry import trace
from skills.confluence import calculate_confluence
from mcp_servers.market_data import get_stock_quote, get_spy_change

tracer = trace.get_tracer("test-flow")


async def test_market_data():
    """Test the MCP market data tools."""
    print("\n" + "=" * 60)
    print("  TEST 1: Market Data MCP Tools (Finnhub)")
    print("=" * 60)

    with tracer.start_as_current_span("test_market_data") as span:
        # Test get_stock_quote
        print("\n[MCP] Fetching NVDA quote...")
        nvda_quote = await get_stock_quote.ainvoke("NVDA")
        print(f"  NVDA: ${nvda_quote['price']:.2f} (H: ${nvda_quote['high']:.2f}, L: ${nvda_quote['low']:.2f})")
        print(f"  Change: {nvda_quote['change_percent']:.2f}%")
        span.set_attribute("nvda.price", nvda_quote['price'])

        # Test get_spy_change
        print("\n[MCP] Fetching SPY for market context...")
        spy_data = await get_spy_change.ainvoke("")
        print(f"  SPY: ${spy_data['price']:.2f} ({spy_data['change_percent']:+.2f}%)")
        print(f"  Market Sentiment: {spy_data['market_sentiment']}")
        span.set_attribute("spy.change", spy_data['change_percent'])

        return nvda_quote, spy_data


def test_confluence_skill(nvda_quote: dict, spy_data: dict):
    """Test the confluence calculation skill."""
    print("\n" + "=" * 60)
    print("  TEST 2: Confluence Skill (V2.1 Calculator)")
    print("=" * 60)

    with tracer.start_as_current_span("test_confluence") as span:
        # Simulate user input parameters
        params = {
            "ticker": "NVDA",
            "price": nvda_quote['price'],
            "high": nvda_quote['high'],
            "low": nvda_quote['low'],
            "rsi": 35,  # User provided
            "volume_ratio": 1.2,  # User provided
            "vwap_position": "touching",  # User provided
            "trend": "up",  # User provided
            "support": nvda_quote['low'],  # Use day low as support
            "spy_change": spy_data['change_percent'],
            "has_catalyst": False
        }

        print(f"\n[SKILL] Calculating confluence for {params['ticker']}...")
        print(f"  Input params:")
        print(f"    Price: ${params['price']:.2f}")
        print(f"    RSI: {params['rsi']}")
        print(f"    Volume: {params['volume_ratio']}x")
        print(f"    VWAP: {params['vwap_position']}")
        print(f"    Support: ${params['support']:.2f}")
        print(f"    SPY: {params['spy_change']:+.2f}%")

        # Call the skill
        result = calculate_confluence.invoke(params)

        span.set_attribute("confluence.score", result['score'])
        span.set_attribute("confluence.verdict", result['verdict'])

        return result


def display_result(result: dict):
    """Display the analysis result."""
    print("\n" + "=" * 60)
    print("  ANALYSIS RESULT")
    print("=" * 60)

    verdict_emoji = {
        "LONG_SCALP": "🚀",
        "WAIT": "⏳",
        "NO_TRADE": "🛑"
    }

    print(f"\n  {verdict_emoji.get(result['verdict'], '❓')} {result['verdict']} ({result['confidence']})")
    print(f"  Score: {result['score']}/10")
    print(f"  Reasoning: {result['reasoning']}")

    print("\n  Factors:")
    for factor in result['factors']:
        score_color = "✅" if factor['score'] >= 1.5 else "⚡" if factor['score'] >= 0.5 else "❌"
        print(f"    {score_color} {factor['name']}: {factor['note']} (+{factor['score']})")

    if result['verdict'] == "LONG_SCALP":
        setup = result['trade_setup']
        print(f"\n  Trade Setup:")
        print(f"    Entry:  ${setup['entry']}")
        print(f"    Stop:   ${setup['stop']} 🔴")
        print(f"    T1:     ${setup['target1']} 🟢")
        print(f"    T2:     ${setup['target2']} 🟢")
        print(f"    R:R:    {setup['risk_reward']}")
        if setup['is_high_beta']:
            print(f"    ⚠️  High-Beta: Use {setup['position_size']} position size")


async def main():
    print("\n" + "🔥" * 30)
    print("  SCALP MVP - Test Flow")
    print("  Testing: MCP Tools → Skills → Phoenix Traces")
    print("🔥" * 30)

    with tracer.start_as_current_span("scalp_analysis_flow") as root_span:
        root_span.set_attribute("test.type", "integration")

        # Test 1: Market Data
        nvda_quote, spy_data = await test_market_data()

        # Test 2: Confluence Calculation
        result = test_confluence_skill(nvda_quote, spy_data)

        # Display Result
        display_result(result)

        root_span.set_attribute("result.verdict", result['verdict'])
        root_span.set_attribute("result.score", result['score'])

    print("\n" + "=" * 60)
    print("  ✅ TEST COMPLETE")
    print("=" * 60)
    print(f"\n  View traces at: http://localhost:6006")
    print(f"  Project: scalp-mvp-test")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
