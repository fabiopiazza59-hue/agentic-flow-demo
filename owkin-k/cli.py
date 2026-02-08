#!/usr/bin/env python3
"""
MCP Gateway CLI - Create and manage MCP servers.

Usage:
    mcp-gateway create-mcp --name "My Server" --template base-python
    mcp-gateway register --url http://localhost:8004
    mcp-gateway list-tools
    mcp-gateway list-servers
"""

import os
import re
import json
import shutil
import httpx
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Default gateway URL
GATEWAY_URL = os.getenv("MCP_GATEWAY_GATEWAY", "http://localhost:8000")
API_KEY = os.getenv("MCP_GATEWAY_API_KEY", "ok-demo-key")

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "mcp-templates"


def api_call(method: str, endpoint: str, data: dict = None):
    """Make an API call to the gateway."""
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    url = f"{GATEWAY_URL}{endpoint}"

    with httpx.Client(timeout=30.0) as client:
        if method == "GET":
            response = client.get(url, headers=headers)
        elif method == "POST":
            response = client.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unknown method: {method}")

    return response.json()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """MCP Gateway CLI - Create and manage MCP servers."""
    pass


@cli.command("create-mcp")
@click.option("--name", "-n", required=True, help="Server name (e.g., 'My LIMS Server')")
@click.option("--template", "-t", default="base-python", help="Template to use (base-python)")
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--port", "-p", default=8004, help="Server port")
@click.option("--description", "-d", default="", help="Server description")
def create_mcp(name: str, template: str, output: str, port: int, description: str):
    """Create a new MCP server from a template."""
    console.print(Panel.fit(f"[bold blue]Creating MCP Server: {name}[/bold blue]"))

    # Generate server ID from name
    server_id = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

    # Template directory
    template_dir = TEMPLATES_DIR / template
    if not template_dir.exists():
        console.print(f"[red]Error: Template '{template}' not found[/red]")
        console.print(f"Available templates: {', '.join(t.name for t in TEMPLATES_DIR.iterdir() if t.is_dir())}")
        return

    # Output directory
    output_path = Path(output) / server_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Template variables
    variables = {
        "server_name": name,
        "server_id": server_id,
        "description": description or f"MCP server for {name}",
        "port": str(port),
        "tool_1_name": "process_data",
        "tool_1_description": "Process input data and return results",
        "tool_2_name": "get_resource",
        "tool_2_description": "Get a resource by ID",
    }

    # Process template files
    console.print("\n[cyan]Generating files:[/cyan]")
    for template_file in template_dir.glob("*.template"):
        # Read template
        content = template_file.read_text()

        # Replace variables
        for key, value in variables.items():
            content = content.replace("{{" + key + "}}", value)

        # Write output file
        output_file = output_path / template_file.stem
        output_file.write_text(content)
        console.print(f"  [green]✓[/green] {output_file}")

    # Create __init__.py
    (output_path / "__init__.py").write_text(f'"""MCP Server: {name}"""\n')

    console.print(f"\n[green]✓ Server created at: {output_path}[/green]")

    console.print(Panel.fit(f"""
[bold]Next Steps:[/bold]

1. [cyan]cd {output_path}[/cyan]

2. Install dependencies:
   [cyan]pip install -r requirements.txt[/cyan]

3. Customize your tools in [cyan]server.py[/cyan]

4. Run locally:
   [cyan]python server.py[/cyan]

5. Register with MCP Gateway:
   [cyan]mcp-gateway register --url http://localhost:{port}[/cyan]
""", title="Quick Start"))


@cli.command("register")
@click.option("--url", "-u", required=True, help="MCP server URL")
@click.option("--name", "-n", default=None, help="Server name (auto-detected if not provided)")
@click.option("--id", "server_id", default=None, help="Server ID (auto-generated if not provided)")
def register(url: str, name: str, server_id: str):
    """Register an MCP server with MCP Gateway Gateway."""
    console.print(f"[cyan]Registering server at {url}...[/cyan]")

    # Try to discover server info
    try:
        with httpx.Client(timeout=10.0) as client:
            # Get health
            health = client.get(f"{url}/health")
            if health.status_code != 200:
                console.print(f"[yellow]Warning: Server health check failed[/yellow]")

            # Get tools
            tools_resp = client.get(f"{url}/tools")
            tools_data = tools_resp.json() if tools_resp.status_code == 200 else {"tools": []}

    except Exception as e:
        console.print(f"[yellow]Warning: Could not connect to server: {e}[/yellow]")
        tools_data = {"tools": []}

    # Generate defaults
    if not server_id:
        server_id = url.split(":")[-1].replace("/", "_") + "_server"
    if not name:
        name = f"Server at {url}"

    # Register with gateway
    try:
        result = api_call("POST", "/mcp/servers", {
            "id": server_id,
            "name": name,
            "url": url,
            "description": f"MCP server registered from {url}",
            "auto_discover": True
        })

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]✓ Server registered successfully![/green]")
            console.print(f"  ID: {server_id}")
            console.print(f"  Name: {name}")
            console.print(f"  Tools discovered: {result.get('tools_discovered', 0)}")

    except Exception as e:
        console.print(f"[red]Error registering server: {e}[/red]")


@cli.command("list-servers")
def list_servers():
    """List all registered MCP servers."""
    try:
        result = api_call("GET", "/mcp/servers")
        servers = result.get("servers", [])

        if not servers:
            console.print("[yellow]No servers registered[/yellow]")
            return

        table = Table(title="MCP Servers")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("URL")
        table.add_column("Tools", justify="right")
        table.add_column("Health", justify="center")

        for s in servers:
            health_icon = {
                "healthy": "[green]●[/green]",
                "unhealthy": "[red]●[/red]",
                "unknown": "[yellow]●[/yellow]"
            }.get(s.get("health_status", "unknown"), "[yellow]●[/yellow]")

            table.add_row(
                s["id"],
                s["name"],
                s["url"],
                str(len(s.get("tools", []))),
                health_icon
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("list-tools")
def list_tools():
    """List all available tools across all servers."""
    try:
        result = api_call("GET", "/mcp/tools")
        tools = result.get("tools", [])

        if not tools:
            console.print("[yellow]No tools available[/yellow]")
            return

        table = Table(title="Available Tools")
        table.add_column("Tool", style="cyan")
        table.add_column("Server", style="green")
        table.add_column("Description")

        for t in tools:
            table.add_row(
                t["full_name"],
                t["server_name"],
                t["description"][:60] + "..." if len(t["description"]) > 60 else t["description"]
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("list-skills")
def list_skills():
    """List all available skills."""
    try:
        result = api_call("GET", "/skills")
        skills = result.get("skills", [])

        if not skills:
            console.print("[yellow]No skills available[/yellow]")
            return

        table = Table(title="Skills")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Description")
        table.add_column("Tools")

        for s in skills:
            table.add_row(
                s["id"],
                s["name"],
                s["description"][:40] + "..." if len(s["description"]) > 40 else s["description"],
                ", ".join(s.get("tools_available", []))
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("test")
@click.argument("server_url")
def test_server(server_url: str):
    """Test an MCP server before registering."""
    console.print(f"[cyan]Testing server at {server_url}...[/cyan]\n")

    with httpx.Client(timeout=10.0) as client:
        # Health check
        try:
            health = client.get(f"{server_url}/health")
            if health.status_code == 200:
                console.print(f"[green]✓[/green] Health check passed")
            else:
                console.print(f"[red]✗[/red] Health check failed: {health.status_code}")
        except Exception as e:
            console.print(f"[red]✗[/red] Health check failed: {e}")
            return

        # Tools discovery
        try:
            tools = client.get(f"{server_url}/tools")
            if tools.status_code == 200:
                data = tools.json()
                tool_list = data.get("tools", [])
                console.print(f"[green]✓[/green] Tools endpoint accessible ({len(tool_list)} tools)")

                for t in tool_list:
                    console.print(f"    - {t.get('name', 'unknown')}: {t.get('description', '')[:50]}")
            else:
                console.print(f"[yellow]![/yellow] Tools endpoint returned: {tools.status_code}")
        except Exception as e:
            console.print(f"[red]✗[/red] Tools discovery failed: {e}")

        # Test tool call
        try:
            tools_data = client.get(f"{server_url}/tools").json()
            if tools_data.get("tools"):
                first_tool = tools_data["tools"][0]
                console.print(f"\n[cyan]Testing tool: {first_tool['name']}[/cyan]")

                call_result = client.post(f"{server_url}/call", json={
                    "tool": first_tool["name"],
                    "arguments": {}
                })

                if call_result.status_code == 200:
                    console.print(f"[green]✓[/green] Tool call successful")
                    result = call_result.json()
                    console.print(Syntax(json.dumps(result, indent=2)[:500], "json"))
                else:
                    console.print(f"[yellow]![/yellow] Tool call returned: {call_result.status_code}")
        except Exception as e:
            console.print(f"[yellow]![/yellow] Tool call test skipped: {e}")

    console.print("\n[green]Server is ready to register![/green]")
    console.print(f"Run: [cyan]mcp-gateway register --url {server_url}[/cyan]")


@cli.command("unregister")
@click.argument("server_id")
def unregister(server_id: str):
    """Unregister an MCP server."""
    try:
        result = api_call("DELETE", f"/mcp/servers/{server_id}")

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]✓ Server '{server_id}' unregistered[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("info")
def gateway_info():
    """Show gateway information."""
    try:
        result = api_call("GET", "/info")
        console.print(Panel.fit(f"""
[bold]MCP Gateway Gateway[/bold]

Version: {result.get('version', 'unknown')}
MCP Servers: {result.get('mcp_servers', 0)}
Skills: {result.get('skills', 0)}
Agents: {result.get('agents', 0)}

Gateway URL: {GATEWAY_URL}
""", title="Gateway Info"))

    except Exception as e:
        console.print(f"[red]Error connecting to gateway: {e}[/red]")
        console.print(f"Is the gateway running at {GATEWAY_URL}?")


if __name__ == "__main__":
    cli()
