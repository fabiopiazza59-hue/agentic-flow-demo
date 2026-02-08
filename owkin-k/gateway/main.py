"""
MCP Gateway Gateway - Central entry point for all MCP operations.

Routes tool calls to registered MCP servers, handles auth, and logs all activity.
"""

import os
import sys
import json
import time
import asyncio
import httpx
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Any, AsyncGenerator

from .auth import get_current_tenant, get_optional_tenant, Tenant, API_KEYS
from .registry import (
    mcp_registry, skill_registry, agent_registry, workflow_registry,
    MCPServer, MCPTool, Skill, AgentConfig, Workflow, WorkflowNode
)

# Add parent to path for agent import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Audit log (in-memory for demo)
audit_log: list[dict] = []

# Background task handle
health_check_task = None


def log_call(tenant_id: str, tool_name: str, latency_ms: float, success: bool):
    """Log a tool call for audit and metering."""
    audit_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "tenant_id": tenant_id,
        "tool_name": tool_name,
        "latency_ms": latency_ms,
        "success": success
    })
    # Keep only last 1000 entries
    if len(audit_log) > 1000:
        audit_log.pop(0)


async def health_check_loop():
    """Background task to check MCP server health every 30 seconds."""
    while True:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                for server in mcp_registry.list_all():
                    try:
                        response = await client.get(f"{server.url}/health")
                        if response.status_code == 200:
                            server.health_status = "healthy"
                        else:
                            server.health_status = "unhealthy"
                    except Exception:
                        server.health_status = "unhealthy"
        except Exception as e:
            print(f"[Health Check] Error: {e}")

        await asyncio.sleep(30)


async def auto_discover_tools(server: MCPServer) -> list[MCPTool]:
    """Auto-discover tools from an MCP server."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{server.url}/tools")
            if response.status_code == 200:
                data = response.json()
                tools = []
                for t in data.get("tools", []):
                    tools.append(MCPTool(
                        name=t.get("name", "unknown"),
                        description=t.get("description", ""),
                        parameters=t.get("parameters", {})
                    ))
                return tools
    except Exception as e:
        print(f"[Auto-discovery] Failed for {server.url}: {e}")
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize registries with MCP MCP servers on startup."""
    global health_check_task

    print("\n" + "=" * 60)
    print("  MCP GATEWAY - Agentic AI Platform")
    print("=" * 60)

    # Register MCP MCP servers
    default_servers = [
        MCPServer(
            id="pathology",
            name="Pathology Engine",
            url="http://localhost:8001",
            description="AI-powered pathology analysis: inference, metadata, annotations",
            tools=[
                MCPTool(name="run_inference", description="Run AI inference on a pathology slide", parameters={"slide_id": "string", "model": "string"}),
                MCPTool(name="get_slide_metadata", description="Get metadata for a slide", parameters={"slide_id": "string"}),
                MCPTool(name="get_annotations", description="Get annotations for a slide", parameters={"slide_id": "string"}),
            ],
            tenant_id="global"
        ),
        MCPServer(
            id="scoring",
            name="Scoring Service",
            url="http://localhost:8002",
            description="Model evaluation and scoring: AUC, baselines, metrics",
            tools=[
                MCPTool(name="compute_auc", description="Compute AUC for model predictions", parameters={"predictions": "array", "labels": "array"}),
                MCPTool(name="compare_baselines", description="Compare model against baselines", parameters={"model_id": "string", "dataset_id": "string"}),
                MCPTool(name="get_metrics", description="Get all metrics for a model", parameters={"model_id": "string"}),
            ],
            tenant_id="global"
        ),
        MCPServer(
            id="datalake",
            name="Data Lake",
            url="http://localhost:8003",
            description="Query datasets, list slides, access curated data",
            tools=[
                MCPTool(name="query_datasets", description="Search available datasets", parameters={"query": "string", "filters": "object"}),
                MCPTool(name="list_slides", description="List slides in a dataset", parameters={"dataset_id": "string", "limit": "integer"}),
                MCPTool(name="get_cohort", description="Get patient cohort data", parameters={"cohort_id": "string"}),
            ],
            tenant_id="global"
        ),
    ]

    for server in default_servers:
        mcp_registry.register(server)
        print(f"[MCP] Registered: {server.name} ({len(server.tools)} tools)")

    # Load default skills
    default_skills = [
        Skill(
            id="pathology-qa",
            name="Pathology Q&A",
            description="Answer questions about pathology slides and analysis",
            content=open_skill_file("pathology-qa"),
            tools_available=["pathology.*"],
            tenant_id="global"
        ),
        Skill(
            id="model-evaluation",
            name="Model Evaluation",
            description="Evaluate and compare AI models",
            content=open_skill_file("model-evaluation"),
            tools_available=["scoring.*", "datalake.*"],
            tenant_id="global"
        ),
        Skill(
            id="report-generation",
            name="Report Generation",
            description="Generate analysis reports",
            content=open_skill_file("report-generation"),
            tools_available=["pathology.*", "datalake.*"],
            tenant_id="global"
        ),
    ]

    for skill in default_skills:
        skill_registry.register(skill)
        print(f"[Skill] Registered: {skill.name}")

    # Start health check background task
    health_check_task = asyncio.create_task(health_check_loop())
    print("[Health] Background health checks started (30s interval)")

    print("=" * 60)
    print(f"[Gateway] Starting on http://0.0.0.0:8000")
    print(f"[Gateway] API Docs: http://0.0.0.0:8000/docs")
    print(f"[Gateway] Console: http://0.0.0.0:8000/console")
    print("=" * 60 + "\n")

    yield

    # Cleanup
    if health_check_task:
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass
    print("\n[Gateway] Shutting down...")


def open_skill_file(skill_id: str) -> str:
    """Load skill markdown content."""
    try:
        with open(f"skills/{skill_id}.md", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"# {skill_id}\n\nSkill content not found."


app = FastAPI(
    title="MCP Gateway Gateway",
    description="Agentic AI Platform for Life Sciences",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for console UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health & Info
# ============================================================

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "mcp-gateway-gateway"}


@app.get("/info")
async def info():
    return {
        "name": "MCP Gateway",
        "version": "1.0.0",
        "description": "Agentic AI Platform for Life Sciences",
        "mcp_servers": len(mcp_registry.list_all()),
        "skills": len(skill_registry.list_all()),
        "agents": len(agent_registry.list_all()),
    }


# ============================================================
# MCP Server Registry
# ============================================================

class RegisterServerRequest(BaseModel):
    id: str
    name: str
    url: str
    description: str = ""
    tools: list[dict] = Field(default_factory=list)
    auto_discover: bool = True  # Auto-discover tools from server


@app.get("/mcp/servers")
async def list_mcp_servers(tenant: Optional[Tenant] = Depends(get_optional_tenant)):
    """List all registered MCP servers."""
    tenant_id = tenant.tenant_id if tenant else None
    servers = mcp_registry.list_all(tenant_id)
    return {"servers": [s.model_dump() for s in servers]}


@app.post("/mcp/servers")
async def register_mcp_server(request: RegisterServerRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Register a new MCP server with optional auto-discovery."""
    server = MCPServer(
        id=request.id,
        name=request.name,
        url=request.url,
        description=request.description,
        tools=[MCPTool(**t) for t in request.tools] if request.tools else [],
        tenant_id=tenant.tenant_id
    )

    # Auto-discover tools if enabled and no tools provided
    if request.auto_discover and not request.tools:
        discovered_tools = await auto_discover_tools(server)
        if discovered_tools:
            server.tools = discovered_tools
            print(f"[Auto-discovery] Found {len(discovered_tools)} tools for {server.name}")

    mcp_registry.register(server)
    return {"status": "registered", "server": server.model_dump(), "tools_discovered": len(server.tools)}


@app.post("/mcp/servers/{server_id}/discover")
async def discover_server_tools(server_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Re-discover tools from an MCP server."""
    server = mcp_registry.get(server_id)
    if not server:
        raise HTTPException(404, "Server not found")
    if server.tenant_id not in (tenant.tenant_id, "global"):
        raise HTTPException(403, "Not authorized")

    discovered_tools = await auto_discover_tools(server)
    if discovered_tools:
        server.tools = discovered_tools
        mcp_registry.register(server)
        return {"status": "discovered", "tools": len(discovered_tools), "server": server.model_dump()}
    else:
        return {"status": "no_tools_found", "tools": 0}


@app.delete("/mcp/servers/{server_id}")
async def unregister_mcp_server(server_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Unregister an MCP server."""
    server = mcp_registry.get(server_id)
    if not server:
        raise HTTPException(404, "Server not found")
    if server.tenant_id not in (tenant.tenant_id, "global"):
        raise HTTPException(403, "Not authorized to delete this server")
    if server.tenant_id == "global":
        raise HTTPException(403, "Cannot delete MCP global servers")

    mcp_registry.unregister(server_id)
    return {"status": "unregistered", "server_id": server_id}


@app.get("/mcp/tools")
async def list_all_tools(tenant: Optional[Tenant] = Depends(get_optional_tenant)):
    """List all tools across all MCP servers."""
    tenant_id = tenant.tenant_id if tenant else None
    tools = mcp_registry.get_all_tools(tenant_id)
    return {"tools": tools}


# ============================================================
# MCP Tool Invocation (JSON-RPC style)
# ============================================================

class ToolCallRequest(BaseModel):
    server_id: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)


class ToolCallResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float


@app.post("/mcp/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Call a tool on an MCP server."""
    start_time = time.time()

    server = mcp_registry.get(request.server_id)
    if not server:
        raise HTTPException(404, f"Server {request.server_id} not found")

    # Check tenant access
    if server.tenant_id not in (tenant.tenant_id, "global"):
        raise HTTPException(403, "Not authorized to access this server")

    # Check tool exists
    tool_names = [t.name for t in server.tools]
    if request.tool_name not in tool_names:
        raise HTTPException(404, f"Tool {request.tool_name} not found on server {request.server_id}")

    # Call the MCP server
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{server.url}/call",
                json={
                    "tool": request.tool_name,
                    "arguments": request.arguments
                }
            )
            response.raise_for_status()
            result = response.json()

        latency = (time.time() - start_time) * 1000
        log_call(tenant.tenant_id, f"{request.server_id}.{request.tool_name}", latency, True)

        return ToolCallResponse(success=True, result=result, latency_ms=latency)

    except httpx.HTTPError as e:
        latency = (time.time() - start_time) * 1000
        log_call(tenant.tenant_id, f"{request.server_id}.{request.tool_name}", latency, False)
        return ToolCallResponse(success=False, error=str(e), latency_ms=latency)


# ============================================================
# Skills Registry
# ============================================================

class CreateSkillRequest(BaseModel):
    id: str
    name: str
    description: str = ""
    content: str
    tools_available: list[str] = Field(default_factory=list)


@app.get("/skills")
async def list_skills(tenant: Optional[Tenant] = Depends(get_optional_tenant)):
    """List all skills."""
    tenant_id = tenant.tenant_id if tenant else None
    skills = skill_registry.list_all(tenant_id)
    return {"skills": [s.model_dump() for s in skills]}


@app.get("/skills/{skill_id}")
async def get_skill(skill_id: str, tenant: Optional[Tenant] = Depends(get_optional_tenant)):
    """Get a skill by ID."""
    skill = skill_registry.get(skill_id)
    if not skill:
        raise HTTPException(404, "Skill not found")
    return skill.model_dump()


@app.post("/skills")
async def create_skill(request: CreateSkillRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Create a new skill."""
    skill = Skill(
        id=request.id,
        name=request.name,
        description=request.description,
        content=request.content,
        tools_available=request.tools_available,
        tenant_id=tenant.tenant_id
    )
    skill_registry.register(skill)
    return {"status": "created", "skill": skill.model_dump()}


@app.put("/skills/{skill_id}")
async def update_skill(skill_id: str, request: CreateSkillRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Update a skill."""
    existing = skill_registry.get(skill_id)
    if not existing:
        raise HTTPException(404, "Skill not found")
    if existing.tenant_id != tenant.tenant_id and existing.tenant_id != "global":
        raise HTTPException(403, "Not authorized to update this skill")

    skill = Skill(
        id=skill_id,
        name=request.name,
        description=request.description,
        content=request.content,
        tools_available=request.tools_available,
        tenant_id=existing.tenant_id,
        version=f"{float(existing.version) + 0.1:.1f}"
    )
    skill_registry.register(skill)
    return {"status": "updated", "skill": skill.model_dump()}


@app.delete("/skills/{skill_id}")
async def delete_skill(skill_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Delete a skill."""
    existing = skill_registry.get(skill_id)
    if not existing:
        raise HTTPException(404, "Skill not found")
    if existing.tenant_id == "global":
        raise HTTPException(403, "Cannot delete global skills")
    if existing.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to delete this skill")

    skill_registry.delete(skill_id)
    return {"status": "deleted", "skill_id": skill_id}


# ============================================================
# Agents Registry & Execution
# ============================================================

class CreateAgentRequest(BaseModel):
    id: str
    name: str
    description: str = ""
    model: str = "claude-sonnet-4-20250514"
    assigned_skills: list[str] = Field(default_factory=list)
    authorized_tools: list[str] = Field(default_factory=list)
    max_iterations: int = 5
    token_budget: int = 4000


class RunAgentRequest(BaseModel):
    query: str
    stream: bool = False


@app.get("/agents")
async def list_agents(tenant: Tenant = Depends(get_current_tenant)):
    """List all agents for tenant."""
    agents = agent_registry.list_all(tenant.tenant_id)
    return {"agents": [a.model_dump() for a in agents]}


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Get an agent by ID."""
    agent = agent_registry.get(agent_id)
    if not agent:
        raise HTTPException(404, "Agent not found")
    if agent.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to access this agent")
    return agent.model_dump()


@app.post("/agents")
async def create_agent(request: CreateAgentRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Create a new agent."""
    agent = AgentConfig(
        id=request.id,
        name=request.name,
        description=request.description,
        model=request.model,
        assigned_skills=request.assigned_skills,
        authorized_tools=request.authorized_tools,
        max_iterations=request.max_iterations,
        token_budget=request.token_budget,
        tenant_id=tenant.tenant_id
    )
    agent_registry.register(agent)
    return {"status": "created", "agent": agent.model_dump()}


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Delete an agent."""
    existing = agent_registry.get(agent_id)
    if not existing:
        raise HTTPException(404, "Agent not found")
    if existing.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to delete this agent")

    agent_registry.delete(agent_id)
    return {"status": "deleted", "agent_id": agent_id}


@app.post("/agents/{agent_id}/run")
async def run_agent(agent_id: str, request: RunAgentRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Run an agent with a query."""
    from agent.runtime import MCPAgent
    from agent.config import AgentRuntimeConfig

    agent_config = agent_registry.get(agent_id)
    if not agent_config:
        raise HTTPException(404, "Agent not found")
    if agent_config.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to run this agent")

    # Convert to runtime config
    runtime_config = AgentRuntimeConfig(
        model=agent_config.model,
        assigned_skills=agent_config.assigned_skills,
        authorized_tools=agent_config.authorized_tools,
        max_iterations=agent_config.max_iterations,
        token_budget=agent_config.token_budget,
        gateway_url="http://localhost:8000",
        api_key="ok-demo-key",
        verbose=False
    )

    agent = MCPAgent(runtime_config)
    result = await agent.run(request.query)

    return {
        "success": result.success,
        "answer": result.answer,
        "steps": [
            {
                "iteration": s.iteration,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation[:500] if s.observation else None
            }
            for s in result.steps
        ],
        "total_iterations": result.total_iterations,
        "tokens_used": result.tokens_used,
        "error": result.error
    }


@app.get("/agents/{agent_id}/stream")
async def stream_agent(agent_id: str, query: str, tenant: Tenant = Depends(get_current_tenant)):
    """Stream agent execution via Server-Sent Events."""
    from agent.runtime import MCPAgentStreaming
    from agent.config import AgentRuntimeConfig

    agent_config = agent_registry.get(agent_id)
    if not agent_config:
        raise HTTPException(404, "Agent not found")
    if agent_config.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to run this agent")

    runtime_config = AgentRuntimeConfig(
        model=agent_config.model,
        assigned_skills=agent_config.assigned_skills,
        authorized_tools=agent_config.authorized_tools,
        max_iterations=agent_config.max_iterations,
        token_budget=agent_config.token_budget,
        gateway_url="http://localhost:8000",
        api_key="ok-demo-key",
        verbose=False
    )

    async def generate():
        agent = MCPAgentStreaming(runtime_config)
        async for event in agent.run_streaming(query):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/playground/run")
async def run_playground(request: RunAgentRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Run a quick agent query in the playground (uses default full-access config)."""
    from agent.runtime import MCPAgent
    from agent.config import AGENT_PRESETS

    config = AGENT_PRESETS["full-access"]
    config.verbose = False

    agent = MCPAgent(config)
    result = await agent.run(request.query)

    return {
        "success": result.success,
        "answer": result.answer,
        "steps": [
            {
                "iteration": s.iteration,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation[:500] if s.observation else None
            }
            for s in result.steps
        ],
        "total_iterations": result.total_iterations,
        "tokens_used": result.tokens_used,
        "error": result.error
    }


@app.get("/playground/stream")
async def stream_playground(query: str, tenant: Tenant = Depends(get_current_tenant)):
    """Stream playground query via Server-Sent Events."""
    from agent.runtime import MCPAgentStreaming
    from agent.config import AGENT_PRESETS

    config = AGENT_PRESETS["full-access"]
    config.verbose = False

    async def generate():
        agent = MCPAgentStreaming(config)
        async for event in agent.run_streaming(query):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================
# Workflows Registry
# ============================================================

class CreateWorkflowRequest(BaseModel):
    id: str
    name: str
    description: str = ""
    nodes: list[dict] = Field(default_factory=list)
    trigger: str = "manual"


@app.get("/workflows")
async def list_workflows(tenant: Tenant = Depends(get_current_tenant)):
    """List all workflows for tenant."""
    workflows = workflow_registry.list_all(tenant.tenant_id)
    return {"workflows": [w.model_dump() for w in workflows]}


@app.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Get a workflow by ID."""
    workflow = workflow_registry.get(workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    if workflow.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to access this workflow")
    return workflow.model_dump()


@app.post("/workflows")
async def create_workflow(request: CreateWorkflowRequest, tenant: Tenant = Depends(get_current_tenant)):
    """Create a new workflow."""
    workflow = Workflow(
        id=request.id,
        name=request.name,
        description=request.description,
        nodes=[WorkflowNode(**n) for n in request.nodes],
        trigger=request.trigger,
        tenant_id=tenant.tenant_id
    )
    workflow_registry.register(workflow)
    return {"status": "created", "workflow": workflow.model_dump()}


@app.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str, tenant: Tenant = Depends(get_current_tenant)):
    """Delete a workflow."""
    existing = workflow_registry.get(workflow_id)
    if not existing:
        raise HTTPException(404, "Workflow not found")
    if existing.tenant_id != tenant.tenant_id:
        raise HTTPException(403, "Not authorized to delete this workflow")

    workflow_registry.delete(workflow_id)
    return {"status": "deleted", "workflow_id": workflow_id}


# ============================================================
# Observability
# ============================================================

@app.get("/audit")
async def get_audit_log(tenant: Tenant = Depends(get_current_tenant), limit: int = 100):
    """Get audit log entries for tenant."""
    entries = [e for e in audit_log if e["tenant_id"] == tenant.tenant_id]
    return {"entries": entries[-limit:]}


@app.get("/metrics")
async def get_metrics(tenant: Tenant = Depends(get_current_tenant)):
    """Get usage metrics for tenant."""
    entries = [e for e in audit_log if e["tenant_id"] == tenant.tenant_id]

    total_calls = len(entries)
    success_calls = len([e for e in entries if e["success"]])
    avg_latency = sum(e["latency_ms"] for e in entries) / total_calls if total_calls > 0 else 0

    # Group by tool
    tool_counts = {}
    for e in entries:
        tool = e["tool_name"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return {
        "total_calls": total_calls,
        "success_rate": success_calls / total_calls if total_calls > 0 else 1.0,
        "avg_latency_ms": avg_latency,
        "calls_by_tool": tool_counts
    }


# ============================================================
# Static Console UI
# ============================================================

@app.get("/console")
async def console_redirect():
    """Redirect to console UI."""
    return FileResponse("console/index.html")


# Mount static files for console
if os.path.exists("console"):
    app.mount("/console", StaticFiles(directory="console", html=True), name="console")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
