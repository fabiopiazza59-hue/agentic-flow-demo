"""
Evaluation API routes.

Run evaluations and view results.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from registry.store import ServerStore, EvaluationStore
from registry.schemas import EvaluationResult
from evaluation.evaluator import ToolEvaluator
from evaluation.test_generator import TestCaseGenerator
from evaluation.metrics import MetricsAggregator
from api.models import RunEvaluationRequest, EvaluationSummaryResponse

router = APIRouter(prefix="/evaluate", tags=["evaluation"])

# Dependencies injected from main.py
server_store: ServerStore = None
evaluation_store: EvaluationStore = None
evaluator: ToolEvaluator = None
test_generator: TestCaseGenerator = None
metrics_aggregator: MetricsAggregator = None


def init_router(
    s_store: ServerStore,
    e_store: EvaluationStore,
):
    """Initialize router with dependencies."""
    global server_store, evaluation_store, evaluator, test_generator, metrics_aggregator
    server_store = s_store
    evaluation_store = e_store
    evaluator = ToolEvaluator()
    test_generator = TestCaseGenerator()
    metrics_aggregator = MetricsAggregator()


@router.post("/{server_id}", response_model=EvaluationSummaryResponse)
async def run_evaluation(server_id: str, request: Optional[RunEvaluationRequest] = None):
    """
    Run evaluation on a server's tools.

    If test_cases are not provided, synthetic test cases will be generated.
    """
    server = server_store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    if not server.tools:
        raise HTTPException(
            status_code=400, detail=f"Server '{server_id}' has no tools to evaluate"
        )

    # Get or generate test cases
    if request and request.test_cases:
        test_cases = request.test_cases
    else:
        num_cases = request.num_synthetic_cases if request else 5
        test_cases = await test_generator.generate_for_server(
            server.tools, cases_per_tool=num_cases
        )

    # Run evaluation for each tool
    all_results = []
    for tool in server.tools:
        # Filter test cases for this tool
        tool_cases = [c for c in test_cases if c.get("expected_tool") == tool.id]
        if not tool_cases:
            # Use all cases if no specific filtering
            tool_cases = test_cases[:3]

        results = await evaluator.evaluate_tool(
            tool_id=tool.id,
            tool_name=tool.name,
            tool_description=tool.description,
            test_cases=tool_cases,
        )
        all_results.extend(results)

    # Aggregate results
    evaluation = metrics_aggregator.aggregate_results(server_id, all_results)

    # Save evaluation
    evaluation_store.save(evaluation)

    return EvaluationSummaryResponse(
        id=evaluation.id,
        server_id=server_id,
        timestamp=evaluation.timestamp,
        avg_quality_score=evaluation.avg_quality_score,
        tool_selection_accuracy=evaluation.tool_selection_accuracy,
        recommendation=evaluation.recommendation,
        reasoning=evaluation.reasoning,
    )


@router.get("/{server_id}/history")
async def get_evaluation_history(server_id: str, limit: int = 10):
    """Get evaluation history for a server."""
    server = server_store.get(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_id}' not found")

    evaluations = evaluation_store.list_for_server(server_id, limit=limit)

    return {
        "server_id": server_id,
        "total_evaluations": len(evaluations),
        "evaluations": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "avg_quality_score": e.avg_quality_score,
                "recommendation": e.recommendation,
            }
            for e in evaluations
        ],
    }


@router.get("/{server_id}/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation(server_id: str, evaluation_id: str):
    """Get detailed evaluation results."""
    evaluation = evaluation_store.get(server_id, evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation '{evaluation_id}' not found for server '{server_id}'",
        )
    return evaluation


@router.get("")
async def list_recent_evaluations(limit: int = 20):
    """List recent evaluations across all servers."""
    evaluations = evaluation_store.list_all(limit=limit)

    return {
        "total": len(evaluations),
        "evaluations": [
            {
                "id": e.id,
                "server_id": e.server_id,
                "timestamp": e.timestamp,
                "avg_quality_score": e.avg_quality_score,
                "recommendation": e.recommendation,
            }
            for e in evaluations
        ],
    }


@router.post("/{server_id}/compare")
async def compare_evaluations(server_id: str, eval_id_1: str, eval_id_2: str):
    """Compare two evaluations for a server."""
    eval1 = evaluation_store.get(server_id, eval_id_1)
    eval2 = evaluation_store.get(server_id, eval_id_2)

    if not eval1:
        raise HTTPException(status_code=404, detail=f"Evaluation '{eval_id_1}' not found")
    if not eval2:
        raise HTTPException(status_code=404, detail=f"Evaluation '{eval_id_2}' not found")

    comparison = metrics_aggregator.compare_evaluations(eval1, eval2)

    return {
        "server_id": server_id,
        "evaluation_1": {"id": eval_id_1, "timestamp": eval1.timestamp},
        "evaluation_2": {"id": eval_id_2, "timestamp": eval2.timestamp},
        "comparison": comparison,
    }
