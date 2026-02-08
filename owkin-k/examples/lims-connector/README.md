# LIMS Connector - Example MCP Server

This is an example MCP server that demonstrates how to create a custom connector for a Laboratory Information Management System (LIMS).

## Quick Start

### 1. Start the server

```bash
cd examples/lims-connector
pip install -r requirements.txt
python server.py
```

### 2. Test the server

```bash
# Health check
curl http://localhost:8004/health

# List tools
curl http://localhost:8004/tools

# Get a sample
curl -X POST http://localhost:8004/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_sample", "arguments": {"sample_id": "SAM-001"}}'
```

### 3. Register with MCP Gateway

Using the CLI:
```bash
mcp-gateway test http://localhost:8004    # Test first
mcp-gateway register --url http://localhost:8004
```

Using the API:
```bash
curl -X POST http://localhost:8000/mcp/servers \
  -H "X-API-Key: ok-demo-key" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "lims",
    "name": "LIMS Connector",
    "url": "http://localhost:8004",
    "description": "Laboratory Information Management System connector"
  }'
```

### 4. Use in the Console

1. Go to http://localhost:8000/console
2. Click "MCP Hub" - you should see the LIMS connector
3. Go to "Playground" and ask:
   - "What samples are available in the LIMS?"
   - "Get details for sample SAM-001"
   - "List running experiments"

## Tools

| Tool | Description |
|------|-------------|
| `get_sample` | Get detailed information about a sample |
| `list_samples` | List all samples with optional status filter |
| `get_experiment` | Get experiment details and progress |
| `list_experiments` | List all experiments |
| `update_sample_status` | Update a sample's status |

## Customization

To connect to a real LIMS:

1. Replace the mock data dictionaries with API calls to your LIMS
2. Add authentication (API keys, OAuth, etc.)
3. Handle error cases from your LIMS API
4. Add more tools as needed

## Creating Your Own Connector

Use the CLI to scaffold a new server:

```bash
mcp-gateway create-mcp --name "My Custom Connector" --port 8005
```

This creates a template you can customize.
