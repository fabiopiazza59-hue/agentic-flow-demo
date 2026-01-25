"""
Scalp MVP - FastAPI Server

Entry point for the agentic scalp trading assistant.
Initializes Phoenix tracing and exposes the /analyze endpoint.

Run: python main.py
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
from orchestrator import get_orchestrator


# Request/Response models
class AnalyzeRequest(BaseModel):
    """Request model for scalp analysis."""
    query: str = Field(
        ...,
        description="Natural language query for scalp trading analysis",
        examples=["Analyze NVDA for a scalp entry", "Check TSLA setup, RSI 35, volume 1.2x"]
    )


class AnalyzeResponse(BaseModel):
    """Response model for scalp analysis."""
    success: bool
    query: str
    response: str
    intent: str | None = None


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize tracing on startup."""
    print("\n" + "=" * 50)
    print("  SCALP MVP - Agentic Trading Assistant")
    print("=" * 50)

    # Initialize Phoenix tracing
    setup_tracing("scalp-mvp")

    print("\n[Server] Starting up...")
    print(f"[Server] API: http://localhost:{os.getenv('PORT', 8000)}")
    print(f"[Server] Docs: http://localhost:{os.getenv('PORT', 8000)}/docs")
    print(f"[Server] Phoenix: {os.getenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')}")
    print("=" * 50 + "\n")

    yield

    print("\n[Server] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Scalp MVP",
    description="Agentic Scalp Trading Assistant using V2.1 Methodology",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the demo UI."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "orchestrator": "ready",
            "scalp_agent": "ready",
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
    Analyze a stock for scalp trading setup.

    The orchestrator will:
    1. Classify your intent
    2. Route to the Scalp Agent
    3. Fetch market data (Finnhub)
    4. Calculate confluence score (V2.1)
    5. Return trade decision

    All steps are traced in Phoenix for observability.
    """
    try:
        # Get the orchestrator
        orchestrator = get_orchestrator()

        # Invoke with user message
        result = await orchestrator.ainvoke({
            "messages": [HumanMessage(content=request.query)],
            "intent": "",
            "analysis_result": None
        })

        # Extract response from messages
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content") and msg.content:
                response_text = msg.content
                break

        return AnalyzeResponse(
            success=True,
            query=request.query,
            response=response_text,
            intent=result.get("intent")
        )

    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
