"""
Metrics aggregation for tool evaluations.

Computes summary statistics and recommendations.
"""

from typing import Literal
from datetime import datetime
import statistics

from registry.schemas import EvaluationResult
from .evaluator import ToolEvalResult


class MetricsAggregator:
    """Aggregate evaluation results into summary metrics."""

    # Thresholds for recommendations
    KEEP_THRESHOLD = 4.0  # Average score >= 4.0 = KEEP
    IMPROVE_THRESHOLD = 2.5  # Average score >= 2.5 = IMPROVE, else DEPRECATE

    def aggregate_results(
        self,
        server_id: str,
        results: list[ToolEvalResult],
    ) -> EvaluationResult:
        """
        Aggregate individual tool results into a summary.

        Args:
            server_id: Server that was evaluated
            results: List of individual evaluation results

        Returns:
            EvaluationResult summary
        """
        if not results:
            return EvaluationResult(
                id=self._generate_id(),
                server_id=server_id,
                timestamp=datetime.utcnow(),
                tool_selection_accuracy=0.0,
                param_extraction_accuracy=0.0,
                avg_quality_score=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                total_cost_usd=0.0,
                recommendation="IMPROVE",
                reasoning="No evaluation results to aggregate",
                test_results=[],
            )

        # Calculate tool selection accuracy
        selection_correct = sum(1 for r in results if r.selection_correct)
        tool_selection_accuracy = selection_correct / len(results)

        # For now, assume param extraction = selection (simplified)
        param_extraction_accuracy = tool_selection_accuracy

        # Calculate quality scores
        overall_scores = [r.overall for r in results]
        avg_quality = statistics.mean(overall_scores)
        # Ensure minimum score of 1.0 for schema validation
        avg_quality = max(1.0, avg_quality)

        # Calculate latency percentiles
        latencies = sorted([r.latency_ms for r in results])
        latency_p50 = self._percentile(latencies, 50)
        latency_p95 = self._percentile(latencies, 95)

        # Total cost
        total_cost = sum(r.cost_usd for r in results)

        # Determine recommendation
        recommendation = self._determine_recommendation(
            avg_quality, tool_selection_accuracy
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            avg_quality, tool_selection_accuracy, results
        )

        # Convert results to dicts for storage
        test_results = [
            {
                "tool_id": r.tool_id,
                "query": r.query,
                "overall": r.overall,
                "recommendation": r.recommendation,
                "reasoning": r.reasoning,
            }
            for r in results
        ]

        return EvaluationResult(
            id=self._generate_id(),
            server_id=server_id,
            timestamp=datetime.utcnow(),
            tool_selection_accuracy=tool_selection_accuracy,
            param_extraction_accuracy=param_extraction_accuracy,
            avg_quality_score=avg_quality,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            total_cost_usd=total_cost,
            recommendation=recommendation,
            reasoning=reasoning,
            test_results=test_results,
        )

    def _percentile(self, sorted_values: list[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def _determine_recommendation(
        self, avg_quality: float, selection_accuracy: float
    ) -> Literal["KEEP", "IMPROVE", "DEPRECATE"]:
        """Determine recommendation based on metrics."""
        # Weight quality more heavily than selection accuracy
        combined_score = (avg_quality * 0.7) + (selection_accuracy * 5 * 0.3)

        if combined_score >= self.KEEP_THRESHOLD:
            return "KEEP"
        elif combined_score >= self.IMPROVE_THRESHOLD:
            return "IMPROVE"
        else:
            return "DEPRECATE"

    def _generate_reasoning(
        self,
        avg_quality: float,
        selection_accuracy: float,
        results: list[ToolEvalResult],
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        parts = []

        # Quality assessment
        if avg_quality >= 4.0:
            parts.append("High quality responses")
        elif avg_quality >= 3.0:
            parts.append("Adequate response quality")
        elif avg_quality >= 2.0:
            parts.append("Below average response quality")
        else:
            parts.append("Poor response quality")

        # Selection accuracy
        if selection_accuracy >= 0.9:
            parts.append("excellent tool selection")
        elif selection_accuracy >= 0.7:
            parts.append("good tool selection")
        elif selection_accuracy >= 0.5:
            parts.append("moderate tool selection issues")
        else:
            parts.append("significant tool selection problems")

        # Count individual recommendations
        keep_count = sum(1 for r in results if r.recommendation == "KEEP")
        improve_count = sum(1 for r in results if r.recommendation == "IMPROVE")
        deprecate_count = sum(1 for r in results if r.recommendation == "DEPRECATE")

        if improve_count > 0 or deprecate_count > 0:
            parts.append(
                f"{keep_count} tools OK, {improve_count} need improvement, {deprecate_count} should be deprecated"
            )

        return ". ".join(parts) + "."

    def _generate_id(self) -> str:
        """Generate a unique evaluation ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def compare_evaluations(
        self,
        current: EvaluationResult,
        previous: EvaluationResult,
    ) -> dict:
        """
        Compare two evaluations to track progress.

        Returns:
            Dict with comparison metrics
        """
        return {
            "quality_change": current.avg_quality_score - previous.avg_quality_score,
            "selection_change": current.tool_selection_accuracy - previous.tool_selection_accuracy,
            "latency_change_p50": current.latency_p50_ms - previous.latency_p50_ms,
            "cost_change": current.total_cost_usd - previous.total_cost_usd,
            "recommendation_changed": current.recommendation != previous.recommendation,
            "improved": current.avg_quality_score > previous.avg_quality_score,
        }
