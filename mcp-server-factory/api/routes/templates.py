"""
Template API routes.

Provides access to pre-built server templates for quick starts.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from registry.schemas import ServerDefinition, ToolDefinition, ToolParameter, ToolImplementation
from registry.store import ServerStore
from api.models import TemplateInfo, CreateFromTemplateRequest, ServerInfo

router = APIRouter(prefix="/templates", tags=["templates"])

# Dependencies injected from main.py
store: ServerStore = None


def init_router(server_store: ServerStore):
    """Initialize router with dependencies."""
    global store
    store = server_store


# Pre-built templates
TEMPLATES = {
    "market_data": {
        "id": "market_data",
        "name": "Market Data Server",
        "description": "Real-time stock quotes and market data",
        "category": "finance",
        "tools": [
            ToolDefinition(
                id="get_stock_quote",
                name="get_stock_quote",
                description="Get current stock quote including price, high, low, and change percentage. Use this when users ask about stock prices.",
                parameters=[
                    ToolParameter(
                        name="symbol",
                        type="string",
                        description="Stock ticker symbol (e.g., NVDA, AMD, TSLA)",
                        required=True,
                    )
                ],
                implementation=ToolImplementation(
                    type="python", handler="handlers.market:fetch_quote"
                ),
            ),
            ToolDefinition(
                id="get_market_summary",
                name="get_market_summary",
                description="Get market summary including major indices and overall sentiment.",
                parameters=[],
                implementation=ToolImplementation(
                    type="python", handler="handlers.market:get_summary"
                ),
            ),
        ],
    },
    "calculator": {
        "id": "calculator",
        "name": "Calculator Server",
        "description": "Mathematical calculations and formulas",
        "category": "utility",
        "tools": [
            ToolDefinition(
                id="calculate",
                name="calculate",
                description="Evaluate a mathematical expression. Supports basic arithmetic, percentages, and common functions.",
                parameters=[
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="Mathematical expression to evaluate (e.g., '2 + 2', '100 * 0.15')",
                        required=True,
                    )
                ],
                implementation=ToolImplementation(
                    type="python", handler="handlers.calc:evaluate"
                ),
            ),
            ToolDefinition(
                id="compound_interest",
                name="compound_interest",
                description="Calculate compound interest over time.",
                parameters=[
                    ToolParameter(
                        name="principal",
                        type="number",
                        description="Initial investment amount",
                        required=True,
                    ),
                    ToolParameter(
                        name="rate",
                        type="number",
                        description="Annual interest rate (decimal, e.g., 0.07 for 7%)",
                        required=True,
                    ),
                    ToolParameter(
                        name="years",
                        type="integer",
                        description="Number of years",
                        required=True,
                    ),
                ],
                implementation=ToolImplementation(
                    type="python", handler="handlers.calc:compound"
                ),
            ),
        ],
    },
    "web_search": {
        "id": "web_search",
        "name": "Web Search Server",
        "description": "Search the web for information",
        "category": "search",
        "tools": [
            ToolDefinition(
                id="search",
                name="search",
                description="Search the web for information. Returns relevant results with titles, snippets, and URLs.",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="Search query",
                        required=True,
                    ),
                    ToolParameter(
                        name="num_results",
                        type="integer",
                        description="Number of results to return",
                        required=False,
                        default=5,
                    ),
                ],
                implementation=ToolImplementation(
                    type="http",
                    endpoint="https://api.example.com/search",
                    method="GET",
                ),
            ),
        ],
    },
    "database": {
        "id": "database",
        "name": "Database Query Server",
        "description": "Query and manage database records",
        "category": "data",
        "tools": [
            ToolDefinition(
                id="query",
                name="query",
                description="Execute a read-only SQL query against the database. Only SELECT statements are allowed.",
                parameters=[
                    ToolParameter(
                        name="sql",
                        type="string",
                        description="SQL SELECT query to execute",
                        required=True,
                    ),
                    ToolParameter(
                        name="limit",
                        type="integer",
                        description="Maximum number of rows to return",
                        required=False,
                        default=100,
                    ),
                ],
                implementation=ToolImplementation(
                    type="python", handler="handlers.db:execute_query"
                ),
            ),
        ],
    },
}


@router.get("", response_model=list[TemplateInfo])
async def list_templates(category: Optional[str] = None):
    """
    List available server templates.

    Optionally filter by category (finance, utility, search, data).
    """
    templates = []
    for template_id, template in TEMPLATES.items():
        if category and template["category"] != category:
            continue
        templates.append(
            TemplateInfo(
                id=template_id,
                name=template["name"],
                description=template["description"],
                category=template["category"],
                tool_count=len(template["tools"]),
                example_tools=[t.id for t in template["tools"]],
            )
        )
    return templates


@router.get("/{template_id}", response_model=TemplateInfo)
async def get_template(template_id: str):
    """Get details about a specific template."""
    template = TEMPLATES.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    return TemplateInfo(
        id=template_id,
        name=template["name"],
        description=template["description"],
        category=template["category"],
        tool_count=len(template["tools"]),
        example_tools=[t.id for t in template["tools"]],
    )


@router.post("/{template_id}/create", response_model=ServerInfo, status_code=201)
async def create_from_template(template_id: str, request: CreateFromTemplateRequest):
    """
    Create a new server from a template.

    The template's tools are copied and can be customized.
    """
    template = TEMPLATES.get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    # Check if server already exists
    if store.exists(request.server_id):
        raise HTTPException(
            status_code=409, detail=f"Server '{request.server_id}' already exists"
        )

    # Create server from template
    server = ServerDefinition(
        id=request.server_id,
        name=request.server_name,
        description=request.description or template["description"],
        source_type="template",
        tools=template["tools"],
        tags=request.tags or [template["category"]],
    )

    store.save(server)
    return ServerInfo.from_definition(server)
