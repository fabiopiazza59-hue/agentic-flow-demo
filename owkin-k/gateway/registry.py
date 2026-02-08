"""
In-memory registries for MCP servers, skills, and agents.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class MCPTool(BaseModel):
    """Tool exposed by an MCP server."""
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)


class MCPServer(BaseModel):
    """Registered MCP server."""
    id: str
    name: str
    url: str
    description: str = ""
    tools: list[MCPTool] = Field(default_factory=list)
    tenant_id: str = "global"  # "global" for MCP servers
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    health_status: str = "unknown"  # "healthy", "unhealthy", "unknown"


class Skill(BaseModel):
    """Skill definition."""
    id: str
    name: str
    description: str
    content: str  # Markdown content
    tools_available: list[str] = Field(default_factory=list)
    tenant_id: str = "global"
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentConfig(BaseModel):
    """Agent configuration."""
    id: str
    name: str
    description: str = ""
    model: str = "claude-sonnet-4-20250514"
    assigned_skills: list[str] = Field(default_factory=list)
    authorized_tools: list[str] = Field(default_factory=list)  # patterns like "pathology.*"
    max_iterations: int = 5
    token_budget: int = 4000
    tenant_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowNode(BaseModel):
    """Node in a workflow DAG."""
    id: str
    type: str  # "agent", "tool", "code", "human_gate"
    config: dict = Field(default_factory=dict)
    next_nodes: list[str] = Field(default_factory=list)


class Workflow(BaseModel):
    """Workflow definition."""
    id: str
    name: str
    description: str = ""
    nodes: list[WorkflowNode] = Field(default_factory=list)
    trigger: str = "manual"  # "manual", "api", "cron", "webhook"
    tenant_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MCPRegistry:
    """Registry for MCP servers."""

    def __init__(self):
        self._servers: dict[str, MCPServer] = {}

    def register(self, server: MCPServer) -> None:
        self._servers[server.id] = server

    def unregister(self, server_id: str) -> bool:
        if server_id in self._servers:
            del self._servers[server_id]
            return True
        return False

    def get(self, server_id: str) -> Optional[MCPServer]:
        return self._servers.get(server_id)

    def list_all(self, tenant_id: Optional[str] = None) -> list[MCPServer]:
        servers = list(self._servers.values())
        if tenant_id:
            servers = [s for s in servers if s.tenant_id in (tenant_id, "global")]
        return servers

    def get_all_tools(self, tenant_id: Optional[str] = None) -> list[dict]:
        """Get all tools across all servers."""
        tools = []
        for server in self.list_all(tenant_id):
            for tool in server.tools:
                tools.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "tool_name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "full_name": f"{server.id}.{tool.name}"
                })
        return tools


class SkillRegistry:
    """Registry for skills."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def list_all(self, tenant_id: Optional[str] = None) -> list[Skill]:
        skills = list(self._skills.values())
        if tenant_id:
            skills = [s for s in skills if s.tenant_id in (tenant_id, "global")]
        return skills

    def delete(self, skill_id: str) -> bool:
        if skill_id in self._skills:
            del self._skills[skill_id]
            return True
        return False


class AgentRegistry:
    """Registry for agents."""

    def __init__(self):
        self._agents: dict[str, AgentConfig] = {}

    def register(self, agent: AgentConfig) -> None:
        self._agents[agent.id] = agent

    def get(self, agent_id: str) -> Optional[AgentConfig]:
        return self._agents.get(agent_id)

    def list_all(self, tenant_id: Optional[str] = None) -> list[AgentConfig]:
        agents = list(self._agents.values())
        if tenant_id:
            agents = [a for a in agents if a.tenant_id == tenant_id]
        return agents

    def delete(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False


class WorkflowRegistry:
    """Registry for workflows."""

    def __init__(self):
        self._workflows: dict[str, Workflow] = {}

    def register(self, workflow: Workflow) -> None:
        self._workflows[workflow.id] = workflow

    def get(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def list_all(self, tenant_id: Optional[str] = None) -> list[Workflow]:
        workflows = list(self._workflows.values())
        if tenant_id:
            workflows = [w for w in workflows if w.tenant_id == tenant_id]
        return workflows

    def delete(self, workflow_id: str) -> bool:
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False


# Global registry instances
mcp_registry = MCPRegistry()
skill_registry = SkillRegistry()
agent_registry = AgentRegistry()
workflow_registry = WorkflowRegistry()
