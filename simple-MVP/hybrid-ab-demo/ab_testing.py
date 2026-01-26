"""
A/B Testing Framework for ReAct vs CodeAct

Runs the same query through both paths and compares:
- Latency (execution time)
- Cost (token usage)
- Accuracy (LLM-as-judge evaluation)

Results are logged to Phoenix for analysis.
"""

import os
import time
import asyncio
from dataclasses import dataclass, asdict
from typing import Optional
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from agents.react_agent import get_react_agent
from agents.codeact_agent import get_codeact_agent

load_dotenv()

# Get tracer for custom spans
tracer = trace.get_tracer("ab-testing")


# ============================================================
# COST CALCULATION
# ============================================================

# Pricing per 1K tokens (as of Jan 2025)
PRICING = {
    "claude-3-5-haiku-latest": {"input": 0.001, "output": 0.005},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in dollars."""
    pricing = PRICING.get(model, {"input": 0.003, "output": 0.015})
    return (input_tokens * pricing["input"] / 1000) + (output_tokens * pricing["output"] / 1000)


# ============================================================
# ACCURACY EVALUATION (LLM-as-Judge)
# ============================================================

JUDGE_PROMPT = """You are evaluating the quality of a financial AI assistant's response.

USER QUERY: {query}

AI RESPONSE: {response}

Rate the response on these criteria (1-5 scale each):

1. RELEVANCE: Does it directly answer the user's question?
2. ACCURACY: Are the facts, numbers, and concepts correct?
3. COMPLETENESS: Does it provide sufficient detail?
4. CLARITY: Is it well-structured and easy to understand?

Respond in this exact format:
RELEVANCE: <score>
ACCURACY: <score>
COMPLETENESS: <score>
CLARITY: <score>
OVERALL: <average of above>
REASONING: <one sentence explanation>"""


async def evaluate_response(query: str, response: str) -> dict:
    """Use LLM-as-judge to evaluate response quality."""
    model = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=256,
    )

    prompt = JUDGE_PROMPT.format(query=query, response=response[:2000])

    try:
        result = await model.ainvoke([HumanMessage(content=prompt)])
        content = result.content

        # Parse scores
        scores = {}
        for line in content.split("\n"):
            for metric in ["RELEVANCE", "ACCURACY", "COMPLETENESS", "CLARITY", "OVERALL"]:
                if line.startswith(f"{metric}:"):
                    try:
                        scores[metric.lower()] = float(line.split(":")[1].strip())
                    except:
                        scores[metric.lower()] = 3.0

        # Extract reasoning
        if "REASONING:" in content:
            scores["reasoning"] = content.split("REASONING:")[1].strip()

        return scores

    except Exception as e:
        return {"error": str(e), "overall": 0}


# ============================================================
# A/B TEST RESULT
# ============================================================

@dataclass
class PathResult:
    """Result from running a single path."""
    path: str
    success: bool
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    code: Optional[str] = None
    error: Optional[str] = None
    evaluation: Optional[dict] = None


@dataclass
class ABTestResult:
    """Combined A/B test result."""
    query: str
    react_result: PathResult
    codeact_result: PathResult
    winner: str  # "react", "codeact", or "tie"
    comparison: dict


# ============================================================
# RUN PATHS
# ============================================================

async def run_react_path(query: str) -> PathResult:
    """Run query through ReAct agent."""
    start_time = time.time()

    with tracer.start_as_current_span("ab_test_react") as span:
        span.set_attribute("ab.path", "react")
        span.set_attribute("ab.query", query[:100])

        try:
            agent = get_react_agent()
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=query)]
            })

            # Extract response
            response = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "content") and msg.content:
                    response = msg.content
                    break

            latency = (time.time() - start_time) * 1000
            input_tokens = estimate_tokens(query)
            output_tokens = estimate_tokens(response)
            model = "claude-3-5-haiku-latest"
            cost = calculate_cost(model, input_tokens, output_tokens)

            # Log to span
            span.set_attribute("ab.latency_ms", latency)
            span.set_attribute("ab.cost_usd", cost)
            span.set_attribute("ab.input_tokens", input_tokens)
            span.set_attribute("ab.output_tokens", output_tokens)
            span.set_attribute("ab.success", True)

            return PathResult(
                path="react",
                success=True,
                response=response,
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model
            )

        except Exception as e:
            span.set_attribute("ab.success", False)
            span.set_attribute("ab.error", str(e))
            return PathResult(
                path="react",
                success=False,
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                model="claude-3-5-haiku-latest",
                error=str(e)
            )


async def run_codeact_path(query: str) -> PathResult:
    """Run query through CodeAct agent."""
    start_time = time.time()

    with tracer.start_as_current_span("ab_test_codeact") as span:
        span.set_attribute("ab.path", "codeact")
        span.set_attribute("ab.query", query[:100])

        try:
            agent = get_codeact_agent()
            result = await agent.run(query)

            latency = (time.time() - start_time) * 1000

            # Build response text
            response_parts = []
            if result.get("explanation"):
                response_parts.append(result["explanation"])
            if result.get("output"):
                response_parts.append(result["output"])
            if result.get("result"):
                response_parts.append(str(result["result"]))
            response = "\n".join(response_parts)

            input_tokens = estimate_tokens(query) + 500  # System prompt
            output_tokens = estimate_tokens(response) + estimate_tokens(result.get("code", ""))
            model = "claude-sonnet-4-20250514"
            cost = calculate_cost(model, input_tokens, output_tokens)

            # Log to span
            span.set_attribute("ab.latency_ms", latency)
            span.set_attribute("ab.cost_usd", cost)
            span.set_attribute("ab.input_tokens", input_tokens)
            span.set_attribute("ab.output_tokens", output_tokens)
            span.set_attribute("ab.success", result.get("success", False))
            span.set_attribute("ab.has_code", bool(result.get("code")))

            return PathResult(
                path="codeact",
                success=result.get("success", False),
                response=response,
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=model,
                code=result.get("code"),
                error=result.get("error")
            )

        except Exception as e:
            span.set_attribute("ab.success", False)
            span.set_attribute("ab.error", str(e))
            return PathResult(
                path="codeact",
                success=False,
                response="",
                latency_ms=(time.time() - start_time) * 1000,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0,
                model="claude-sonnet-4-20250514",
                error=str(e)
            )


# ============================================================
# A/B TEST RUNNER
# ============================================================

async def run_ab_test(query: str, evaluate: bool = True) -> ABTestResult:
    """
    Run the same query through both paths and compare results.

    Args:
        query: The user query to test
        evaluate: Whether to run LLM-as-judge evaluation

    Returns:
        ABTestResult with comparison metrics
    """
    with tracer.start_as_current_span("ab_test_comparison") as span:
        span.set_attribute("ab.query", query[:200])
        span.set_attribute("ab.evaluate", evaluate)

        # Run both paths in parallel
        react_result, codeact_result = await asyncio.gather(
            run_react_path(query),
            run_codeact_path(query)
        )

        # Evaluate responses if requested
        if evaluate and react_result.success and codeact_result.success:
            react_eval, codeact_eval = await asyncio.gather(
                evaluate_response(query, react_result.response),
                evaluate_response(query, codeact_result.response)
            )
            react_result.evaluation = react_eval
            codeact_result.evaluation = codeact_eval

            span.set_attribute("ab.react_score", react_eval.get("overall", 0))
            span.set_attribute("ab.codeact_score", codeact_eval.get("overall", 0))

        # Calculate comparison metrics
        comparison = {
            "latency_diff_ms": codeact_result.latency_ms - react_result.latency_ms,
            "latency_ratio": codeact_result.latency_ms / max(react_result.latency_ms, 1),
            "cost_diff_usd": codeact_result.cost_usd - react_result.cost_usd,
            "cost_ratio": codeact_result.cost_usd / max(react_result.cost_usd, 0.0001),
            "react_faster": react_result.latency_ms < codeact_result.latency_ms,
            "react_cheaper": react_result.cost_usd < codeact_result.cost_usd,
        }

        if evaluate and react_result.evaluation and codeact_result.evaluation:
            react_score = react_result.evaluation.get("overall", 0)
            codeact_score = codeact_result.evaluation.get("overall", 0)
            comparison["react_score"] = react_score
            comparison["codeact_score"] = codeact_score
            comparison["score_diff"] = codeact_score - react_score
            comparison["codeact_more_accurate"] = codeact_score > react_score

        # Determine winner (weighted scoring)
        # Accuracy: 50%, Cost: 30%, Latency: 20%
        react_weighted = 0
        codeact_weighted = 0

        if evaluate and react_result.evaluation and codeact_result.evaluation:
            react_weighted += (react_result.evaluation.get("overall", 0) / 5) * 50
            codeact_weighted += (codeact_result.evaluation.get("overall", 0) / 5) * 50

        # Cost score (lower is better, normalize to 0-1)
        max_cost = max(react_result.cost_usd, codeact_result.cost_usd, 0.001)
        react_weighted += (1 - react_result.cost_usd / max_cost) * 30
        codeact_weighted += (1 - codeact_result.cost_usd / max_cost) * 30

        # Latency score (lower is better)
        max_latency = max(react_result.latency_ms, codeact_result.latency_ms, 1)
        react_weighted += (1 - react_result.latency_ms / max_latency) * 20
        codeact_weighted += (1 - codeact_result.latency_ms / max_latency) * 20

        comparison["react_weighted_score"] = react_weighted
        comparison["codeact_weighted_score"] = codeact_weighted

        if abs(react_weighted - codeact_weighted) < 5:
            winner = "tie"
        elif react_weighted > codeact_weighted:
            winner = "react"
        else:
            winner = "codeact"

        # Log comparison to span
        span.set_attribute("ab.winner", winner)
        span.set_attribute("ab.react_weighted", react_weighted)
        span.set_attribute("ab.codeact_weighted", codeact_weighted)
        span.set_attribute("ab.latency_ratio", comparison["latency_ratio"])
        span.set_attribute("ab.cost_ratio", comparison["cost_ratio"])

        return ABTestResult(
            query=query,
            react_result=react_result,
            codeact_result=codeact_result,
            winner=winner,
            comparison=comparison
        )


# ============================================================
# BATCH TESTING
# ============================================================

# Test queries designed to work for BOTH paths
AB_TEST_QUERIES = [
    # Financial calculations (both can handle)
    "Calculate compound interest on $10,000 at 7% for 10 years",
    "What's the future value of $500 monthly investment over 20 years at 8% return?",

    # Analysis tasks (CodeAct should excel)
    "Analyze the probability of reaching $1 million with $100k initial and $1k monthly over 25 years",
    "Compare risk-adjusted returns for a 60/40 vs 80/20 portfolio allocation",

    # Market queries (ReAct should be faster)
    "What's the current price of NVDA and how is the market doing?",
    "Explain what a Sharpe ratio is and why it matters",
]


async def run_batch_ab_test(queries: list = None) -> list[ABTestResult]:
    """Run A/B tests on multiple queries."""
    if queries is None:
        queries = AB_TEST_QUERIES

    results = []
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Testing: {query[:50]}...")
        result = await run_ab_test(query, evaluate=True)
        results.append(result)
        print(f"  Winner: {result.winner}")
        print(f"  React: {result.react_result.latency_ms:.0f}ms, ${result.react_result.cost_usd:.4f}")
        print(f"  CodeAct: {result.codeact_result.latency_ms:.0f}ms, ${result.codeact_result.cost_usd:.4f}")

    return results


def print_summary(results: list[ABTestResult]):
    """Print summary of A/B test results."""
    print("\n" + "=" * 70)
    print("A/B TEST SUMMARY")
    print("=" * 70)

    react_wins = sum(1 for r in results if r.winner == "react")
    codeact_wins = sum(1 for r in results if r.winner == "codeact")
    ties = sum(1 for r in results if r.winner == "tie")

    print(f"\nWinners: ReAct={react_wins}, CodeAct={codeact_wins}, Tie={ties}")

    avg_react_latency = sum(r.react_result.latency_ms for r in results) / len(results)
    avg_codeact_latency = sum(r.codeact_result.latency_ms for r in results) / len(results)
    print(f"\nAvg Latency: ReAct={avg_react_latency:.0f}ms, CodeAct={avg_codeact_latency:.0f}ms")

    total_react_cost = sum(r.react_result.cost_usd for r in results)
    total_codeact_cost = sum(r.codeact_result.cost_usd for r in results)
    print(f"Total Cost: ReAct=${total_react_cost:.4f}, CodeAct=${total_codeact_cost:.4f}")

    if results[0].react_result.evaluation:
        avg_react_score = sum(r.react_result.evaluation.get("overall", 0) for r in results) / len(results)
        avg_codeact_score = sum(r.codeact_result.evaluation.get("overall", 0) for r in results) / len(results)
        print(f"Avg Quality: ReAct={avg_react_score:.2f}/5, CodeAct={avg_codeact_score:.2f}/5")

    print("\n" + "=" * 70)
    print("View detailed traces in Phoenix: http://localhost:6006")
    print("Filter by: ab.path, ab.winner, ab.cost_usd, ab.latency_ms")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from core.tracing import setup_tracing

    # Initialize tracing
    setup_tracing("hybrid-mvp-ab-test")

    print("\n" + "=" * 70)
    print("A/B TEST: ReAct vs CodeAct")
    print("=" * 70)

    # Run single test
    async def main():
        # Single query test
        query = "Calculate the probability of a $50k investment doubling in 10 years with 8% avg return and 15% volatility"

        print(f"\nQuery: {query}\n")
        result = await run_ab_test(query, evaluate=True)

        print(f"\n{'='*50}")
        print(f"REACT PATH (Haiku + Tools)")
        print(f"{'='*50}")
        print(f"Latency: {result.react_result.latency_ms:.0f}ms")
        print(f"Cost: ${result.react_result.cost_usd:.4f}")
        print(f"Response: {result.react_result.response[:300]}...")
        if result.react_result.evaluation:
            print(f"Quality Score: {result.react_result.evaluation.get('overall', 'N/A')}/5")

        print(f"\n{'='*50}")
        print(f"CODEACT PATH (Sonnet + Python)")
        print(f"{'='*50}")
        print(f"Latency: {result.codeact_result.latency_ms:.0f}ms")
        print(f"Cost: ${result.codeact_result.cost_usd:.4f}")
        print(f"Response: {result.codeact_result.response[:300]}...")
        if result.codeact_result.evaluation:
            print(f"Quality Score: {result.codeact_result.evaluation.get('overall', 'N/A')}/5")

        print(f"\n{'='*50}")
        print(f"WINNER: {result.winner.upper()}")
        print(f"{'='*50}")
        print(f"Latency ratio (CodeAct/ReAct): {result.comparison['latency_ratio']:.2f}x")
        print(f"Cost ratio (CodeAct/ReAct): {result.comparison['cost_ratio']:.2f}x")

        print("\n\nView traces in Phoenix: http://localhost:6006")
        print("Project: hybrid-mvp-ab-test")

    asyncio.run(main())
