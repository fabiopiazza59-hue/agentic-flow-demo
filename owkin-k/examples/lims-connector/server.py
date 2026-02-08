"""
LIMS Connector - MCP Server
A sample MCP server for Laboratory Information Management System integration.

This is an example of what a customer would create using the mcp-gateway CLI.

Usage:
    python server.py

Then register with MCP Gateway:
    mcp-gateway register --url http://localhost:8004
"""

import random
from datetime import datetime, timedelta
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LIMS Connector MCP Server")


# Mock LIMS data
SAMPLES = {
    "SAM-001": {"id": "SAM-001", "patient_id": "P-12345", "type": "blood", "status": "processed", "collected_at": "2024-01-15"},
    "SAM-002": {"id": "SAM-002", "patient_id": "P-12346", "type": "tissue", "status": "pending", "collected_at": "2024-01-16"},
    "SAM-003": {"id": "SAM-003", "patient_id": "P-12347", "type": "urine", "status": "processed", "collected_at": "2024-01-17"},
    "SAM-004": {"id": "SAM-004", "patient_id": "P-12348", "type": "blood", "status": "in_progress", "collected_at": "2024-01-18"},
}

EXPERIMENTS = {
    "EXP-001": {"id": "EXP-001", "name": "Gene Expression Analysis", "samples": ["SAM-001", "SAM-002"], "status": "completed"},
    "EXP-002": {"id": "EXP-002", "name": "Protein Quantification", "samples": ["SAM-003"], "status": "running"},
    "EXP-003": {"id": "EXP-003", "name": "Mutation Screening", "samples": ["SAM-004"], "status": "queued"},
}


class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "lims-connector"}


@app.get("/tools")
async def list_tools():
    """List available tools (MCP discovery)."""
    return {
        "tools": [
            {
                "name": "get_sample",
                "description": "Get detailed information about a sample by its ID. Returns sample type, status, patient ID, and collection date.",
                "parameters": {
                    "sample_id": {"type": "string", "description": "The sample identifier (e.g., SAM-001)"}
                }
            },
            {
                "name": "list_samples",
                "description": "List all samples in the LIMS, optionally filtered by status (pending, in_progress, processed).",
                "parameters": {
                    "status": {"type": "string", "description": "Filter by status (optional)"},
                    "limit": {"type": "integer", "description": "Maximum number of samples to return"}
                }
            },
            {
                "name": "get_experiment",
                "description": "Get details about an experiment including its status and associated samples.",
                "parameters": {
                    "experiment_id": {"type": "string", "description": "The experiment identifier (e.g., EXP-001)"}
                }
            },
            {
                "name": "list_experiments",
                "description": "List all experiments in the LIMS with their status and sample count.",
                "parameters": {
                    "status": {"type": "string", "description": "Filter by status (optional)"}
                }
            },
            {
                "name": "update_sample_status",
                "description": "Update the status of a sample (pending, in_progress, processed, failed).",
                "parameters": {
                    "sample_id": {"type": "string", "description": "The sample identifier"},
                    "status": {"type": "string", "description": "New status"}
                }
            }
        ]
    }


@app.post("/call")
async def call_tool(request: ToolCallRequest):
    """Handle tool calls (MCP invocation)."""
    tool = request.tool
    args = request.arguments

    if tool == "get_sample":
        return await get_sample(args.get("sample_id", ""))
    elif tool == "list_samples":
        return await list_samples(args.get("status"), args.get("limit", 10))
    elif tool == "get_experiment":
        return await get_experiment(args.get("experiment_id", ""))
    elif tool == "list_experiments":
        return await list_experiments(args.get("status"))
    elif tool == "update_sample_status":
        return await update_sample_status(args.get("sample_id", ""), args.get("status", ""))
    else:
        return {"error": f"Unknown tool: {tool}"}


async def get_sample(sample_id: str) -> dict:
    """Get detailed information about a sample."""
    if sample_id not in SAMPLES:
        return {"error": f"Sample {sample_id} not found", "available": list(SAMPLES.keys())}

    sample = SAMPLES[sample_id]
    return {
        "sample": sample,
        "metadata": {
            "storage_location": f"Freezer-{random.randint(1, 5)}-Rack-{random.randint(1, 10)}",
            "temperature": f"{random.randint(-80, -70)}°C",
            "volume_ml": round(random.uniform(0.5, 5.0), 2),
            "quality_score": round(random.uniform(0.8, 1.0), 2)
        },
        "experiments": [e["id"] for e in EXPERIMENTS.values() if sample_id in e["samples"]],
        "last_updated": datetime.utcnow().isoformat()
    }


async def list_samples(status: str = None, limit: int = 10) -> dict:
    """List all samples, optionally filtered by status."""
    samples = list(SAMPLES.values())

    if status:
        samples = [s for s in samples if s["status"] == status]

    return {
        "total": len(samples),
        "samples": samples[:limit],
        "filters_applied": {"status": status} if status else None
    }


async def get_experiment(experiment_id: str) -> dict:
    """Get experiment details."""
    if experiment_id not in EXPERIMENTS:
        return {"error": f"Experiment {experiment_id} not found", "available": list(EXPERIMENTS.keys())}

    exp = EXPERIMENTS[experiment_id]
    return {
        "experiment": exp,
        "samples_detail": [SAMPLES.get(s, {"id": s, "status": "unknown"}) for s in exp["samples"]],
        "progress": random.randint(0, 100) if exp["status"] == "running" else (100 if exp["status"] == "completed" else 0),
        "started_at": (datetime.utcnow() - timedelta(days=random.randint(1, 7))).isoformat(),
        "estimated_completion": (datetime.utcnow() + timedelta(days=random.randint(1, 3))).isoformat() if exp["status"] == "running" else None
    }


async def list_experiments(status: str = None) -> dict:
    """List all experiments."""
    experiments = list(EXPERIMENTS.values())

    if status:
        experiments = [e for e in experiments if e["status"] == status]

    return {
        "total": len(experiments),
        "experiments": experiments,
        "status_summary": {
            "completed": len([e for e in EXPERIMENTS.values() if e["status"] == "completed"]),
            "running": len([e for e in EXPERIMENTS.values() if e["status"] == "running"]),
            "queued": len([e for e in EXPERIMENTS.values() if e["status"] == "queued"])
        }
    }


async def update_sample_status(sample_id: str, status: str) -> dict:
    """Update sample status."""
    if sample_id not in SAMPLES:
        return {"error": f"Sample {sample_id} not found"}

    valid_statuses = ["pending", "in_progress", "processed", "failed"]
    if status not in valid_statuses:
        return {"error": f"Invalid status. Must be one of: {valid_statuses}"}

    old_status = SAMPLES[sample_id]["status"]
    SAMPLES[sample_id]["status"] = status

    return {
        "sample_id": sample_id,
        "old_status": old_status,
        "new_status": status,
        "updated_at": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    print("""
============================================================
  LIMS Connector MCP Server
============================================================

This is an example MCP server that connects to a Laboratory
Information Management System.

Server URL: http://localhost:8004
Health:     http://localhost:8004/health
Tools:      http://localhost:8004/tools

To register with MCP Gateway:
  mcp-gateway register --url http://localhost:8004

============================================================
""")
    uvicorn.run(app, host="0.0.0.0", port=8004)
