"""
LLM-as-Judge Evaluator for MCP Tools.

Evaluates tool quality using LLM-based judgment.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from opentelemetry import trace

tracer = trace.get_tracer("mcp-factory-evaluation")


JUDGE_PROMPT = """You are evaluating an MCP tool's response quality.

TOOL: {tool_name}
TOOL DESCRIPTION: {tool_description}
USER QUERY: {query}
TOOL RESPONSE: {response}

Rate the response on these criteria (1-5 scale each):

1. RELEVANCE: Does the tool response directly address the user's need?
2. ACCURACY: Are the returned values correct and well-formatted?
3. COMPLETENESS: Does it provide all necessary information?
4. CLARITY: Is the response structure clear and usable?

Respond in this exact format:
RELEVANCE: <score>
ACCURACY: <score>
COMPLETENESS: <score>
CLARITY: <score>
OVERALL: <average>
RECOMMENDATION: <KEEP|IMPROVE|DEPRECATE>
REASONING: <one sentence explanation>"""


@dataclass
class ToolEvalResult:
    """Result from evaluating a single tool invocation."""

    tool_id: str
    query: str
    tool_selected: str
    selection_correct: bool
    relevance: float
    accuracy: float
    completeness: float
    clarity: float
    overall: float
    latency_ms: float
    cost_usd: float
    recommendation: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ToolEvaluator:
    """Evaluate MCP tools using LLM-as-judge pattern."""

    def __init__(self, model: str = "claude-3-5-haiku-latest"):
        """
        Initialize the evaluator.

        Args:
            model: LLM model to use for evaluation
        """
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.judge = ChatAnthropic(
                model=model,
                api_key=api_key,
                max_tokens=256,
            )
        else:
            self.judge = None

    async def evaluate_tool(
        self,
        tool_id: str,
        tool_name: str,
        tool_description: str,
        test_cases: list[dict],
    ) -> list[ToolEvalResult]:
        """
        Evaluate tool against test cases.

        Args:
            tool_id: Tool identifier
            tool_name: Human-readable tool name
            tool_description: Tool description
            test_cases: List of test cases with query/response pairs

        Returns:
            List of evaluation results
        """
        if not self.judge:
            return [
                ToolEvalResult(
                    tool_id=tool_id,
                    query=case.get("query", ""),
                    tool_selected=tool_id,
                    selection_correct=True,
                    relevance=0,
                    accuracy=0,
                    completeness=0,
                    clarity=0,
                    overall=0,
                    latency_ms=0,
                    cost_usd=0,
                    recommendation="IMPROVE",
                    reasoning="Evaluation skipped: No API key configured",
                )
                for case in test_cases
            ]

        results = []

        with tracer.start_as_current_span("tool_evaluation") as span:
            span.set_attribute("tool_id", tool_id)
            span.set_attribute("num_test_cases", len(test_cases))

            for case in test_cases:
                result = await self._evaluate_case(
                    tool_id, tool_name, tool_description, case
                )
                results.append(result)

            # Log aggregates
            if results:
                avg_overall = sum(r.overall for r in results) / len(results)
                span.set_attribute("avg_overall", avg_overall)

        return results

    async def _evaluate_case(
        self,
        tool_id: str,
        tool_name: str,
        tool_description: str,
        case: dict,
    ) -> ToolEvalResult:
        """Evaluate single test case."""
        query = case.get("query", "")
        response = case.get("response", "")
        expected_tool = case.get("expected_tool", tool_id)

        start_time = time.time()

        prompt = JUDGE_PROMPT.format(
            tool_name=tool_name,
            tool_description=tool_description,
            query=query,
            response=str(response)[:2000],
        )

        try:
            result = await self.judge.ainvoke([HumanMessage(content=prompt)])
            content = result.content

            # Parse scores
            scores = self._parse_scores(content)
            latency = (time.time() - start_time) * 1000

            # Estimate cost (rough)
            input_tokens = len(prompt) // 4
            output_tokens = len(content) // 4
            cost = (input_tokens * 0.001 / 1000) + (output_tokens * 0.005 / 1000)

            return ToolEvalResult(
                tool_id=tool_id,
                query=query,
                tool_selected=tool_id,
                selection_correct=(tool_id == expected_tool),
                relevance=scores.get("relevance", 3.0),
                accuracy=scores.get("accuracy", 3.0),
                completeness=scores.get("completeness", 3.0),
                clarity=scores.get("clarity", 3.0),
                overall=scores.get("overall", 3.0),
                latency_ms=latency,
                cost_usd=cost,
                recommendation=scores.get("recommendation", "KEEP"),
                reasoning=scores.get("reasoning", ""),
            )

        except Exception as e:
            return ToolEvalResult(
                tool_id=tool_id,
                query=query,
                tool_selected=tool_id,
                selection_correct=False,
                relevance=0,
                accuracy=0,
                completeness=0,
                clarity=0,
                overall=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0,
                recommendation="IMPROVE",
                reasoning=f"Evaluation failed: {str(e)}",
            )

    def _parse_scores(self, content: str) -> dict:
        """Parse LLM judge response."""
        scores = {}
        for line in content.split("\n"):
            line = line.strip()
            for metric in ["RELEVANCE", "ACCURACY", "COMPLETENESS", "CLARITY", "OVERALL"]:
                if line.upper().startswith(f"{metric}:"):
                    try:
                        value = line.split(":", 1)[1].strip()
                        # Handle values like "4/5" or just "4"
                        if "/" in value:
                            value = value.split("/")[0]
                        scores[metric.lower()] = float(value)
                    except (ValueError, IndexError):
                        scores[metric.lower()] = 3.0

            if line.upper().startswith("RECOMMENDATION:"):
                rec = line.split(":", 1)[1].strip().upper()
                if rec in ["KEEP", "IMPROVE", "DEPRECATE"]:
                    scores["recommendation"] = rec
                else:
                    scores["recommendation"] = "KEEP"

            if line.upper().startswith("REASONING:"):
                scores["reasoning"] = line.split(":", 1)[1].strip()

        return scores

    async def quick_evaluate(
        self,
        tool_name: str,
        tool_description: str,
        query: str,
        response: str,
    ) -> dict:
        """
        Quick evaluation of a single tool response.

        Returns a simplified evaluation dict.
        """
        result = await self._evaluate_case(
            tool_id="quick_eval",
            tool_name=tool_name,
            tool_description=tool_description,
            case={"query": query, "response": response},
        )

        return {
            "overall": result.overall,
            "relevance": result.relevance,
            "accuracy": result.accuracy,
            "completeness": result.completeness,
            "clarity": result.clarity,
            "recommendation": result.recommendation,
            "reasoning": result.reasoning,
            "latency_ms": result.latency_ms,
        }
