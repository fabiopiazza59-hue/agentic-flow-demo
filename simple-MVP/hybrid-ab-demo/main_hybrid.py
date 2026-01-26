"""
Hybrid MVP - FastAPI Server

Entry point for the hybrid ReAct + CodeAct financial assistant.
Demonstrates A/B comparison between tool-based and code-based agents.

Run: python main_hybrid.py
API: http://localhost:8000
Docs: http://localhost:8000/docs
Phoenix: http://localhost:6006
"""

import os
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# Load environment variables first
load_dotenv()

# Import after env is loaded
from core.tracing import setup_tracing
from hybrid_orchestrator import get_hybrid_orchestrator
from ab_testing import run_ab_test, ABTestResult


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    """Request model for analysis."""
    query: str = Field(
        ...,
        description="Natural language query for financial analysis",
        examples=[
            "What's Apple's stock price?",
            "Run a Monte Carlo simulation for retirement with $100k over 30 years"
        ]
    )


class AnalyzeResponse(BaseModel):
    """Response model with path information for A/B comparison."""
    success: bool
    query: str
    response: str
    path: str  # "react" or "codeact"
    intent: str | None = None
    complexity_score: int | None = None
    execution_time_ms: float | None = None
    code: str | None = None  # Only for CodeAct path


class PathResultModel(BaseModel):
    """Result from a single path."""
    path: str
    success: bool
    response: str
    latency_ms: float
    cost_usd: float
    model: str
    evaluation: dict | None = None
    code: str | None = None


class ABTestResponse(BaseModel):
    """A/B test comparison response."""
    query: str
    react: PathResultModel
    codeact: PathResultModel
    winner: str
    comparison: dict


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tracing on startup."""
    print("\n" + "=" * 60)
    print("  HYBRID MVP - ReAct + CodeAct Demo")
    print("=" * 60)

    # Initialize Phoenix tracing
    setup_tracing("hybrid-mvp")

    print("\n[Server] Starting up...")
    print(f"[Server] API: http://localhost:{os.getenv('PORT', 8000)}")
    print(f"[Server] Docs: http://localhost:{os.getenv('PORT', 8000)}/docs")
    print(f"[Server] Phoenix: {os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')}")
    print("=" * 60 + "\n")

    yield

    print("\n[Server] Shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hybrid MVP",
    description="ReAct + CodeAct Financial Assistant - A/B Comparison Demo",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def root():
    """Serve the demo UI."""
    return FileResponse("static/hybrid_index.html")


@app.get("/health")
async def health():
    """Health check with component status."""
    return {
        "status": "healthy",
        "mode": "hybrid",
        "paths": {
            "react": "ready",
            "codeact": "ready"
        },
        "components": {
            "orchestrator": "ready",
            "react_agent": "ready",
            "codeact_agent": "ready",
            "sandbox": "ready",
            "tracing": "active"
        },
        "config": {
            "anthropic_key": "configured" if os.getenv("ANTHROPIC_API_KEY") else "missing",
            "finnhub_key": "configured" if os.getenv("FINNHUB_API_KEY") else "using_mock",
            "phoenix": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        }
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze a financial query using the hybrid architecture.

    The orchestrator will:
    1. Classify intent and complexity
    2. Route to ReAct (simple) or CodeAct (complex)
    3. Execute the appropriate agent
    4. Return results with path info for comparison

    All steps are traced in Phoenix.
    """
    import time
    start_time = time.time()

    try:
        orchestrator = get_hybrid_orchestrator()

        result = await orchestrator.ainvoke({
            "messages": [HumanMessage(content=request.query)],
            "query": "",
            "intent": "",
            "complexity_score": 0,
            "path": "",
            "result": None
        })

        execution_time = (time.time() - start_time) * 1000

        # Extract response from messages
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                response_text = msg.content
                break

        # Get code if CodeAct was used
        code = None
        agent_result = result.get("result", {})
        if isinstance(agent_result, dict) and agent_result.get("path") == "codeact":
            code = agent_result.get("code")

        return AnalyzeResponse(
            success=True,
            query=request.query,
            response=response_text,
            path=result.get("path", "unknown"),
            intent=result.get("intent"),
            complexity_score=result.get("complexity_score"),
            execution_time_ms=execution_time,
            code=code
        )

    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ab-test", response_model=ABTestResponse)
async def ab_test(request: AnalyzeRequest):
    """
    Run A/B test: same query through BOTH ReAct and CodeAct paths.

    Compares:
    - Latency (execution time)
    - Cost (estimated token cost)
    - Accuracy (LLM-as-judge evaluation)

    All metrics are logged to Phoenix for analysis.
    """
    try:
        result = await run_ab_test(request.query, evaluate=True)

        return ABTestResponse(
            query=request.query,
            react=PathResultModel(
                path="react",
                success=result.react_result.success,
                response=result.react_result.response,
                latency_ms=result.react_result.latency_ms,
                cost_usd=result.react_result.cost_usd,
                model=result.react_result.model,
                evaluation=result.react_result.evaluation,
            ),
            codeact=PathResultModel(
                path="codeact",
                success=result.codeact_result.success,
                response=result.codeact_result.response,
                latency_ms=result.codeact_result.latency_ms,
                cost_usd=result.codeact_result.cost_usd,
                model=result.codeact_result.model,
                evaluation=result.codeact_result.evaluation,
                code=result.codeact_result.code,
            ),
            winner=result.winner,
            comparison=result.comparison
        )

    except Exception as e:
        print(f"[Error] A/B test failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main_hybrid:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
