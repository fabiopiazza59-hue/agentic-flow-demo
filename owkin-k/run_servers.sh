#!/bin/bash
# Run all MCP Gateway servers locally

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "============================================================"
echo "  MCP GATEWAY - Starting All Services"
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check for virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}[Stopping all services...]${NC}"
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start MCP servers in background
echo -e "${GREEN}[Starting]${NC} Pathology Engine on :8001"
python3 -m uvicorn mcp_servers.pathology.server:app --host 0.0.0.0 --port 8001 &

echo -e "${GREEN}[Starting]${NC} Scoring Service on :8002"
python3 -m uvicorn mcp_servers.scoring.server:app --host 0.0.0.0 --port 8002 &

echo -e "${GREEN}[Starting]${NC} Data Lake on :8003"
python3 -m uvicorn mcp_servers.datalake.server:app --host 0.0.0.0 --port 8003 &

# Wait for MCP servers
sleep 2

# Start Gateway
echo -e "${GREEN}[Starting]${NC} Gateway on :8000"
python3 -m uvicorn gateway.main:app --host 0.0.0.0 --port 8000 --reload &

# Wait and show status
sleep 3

echo ""
echo "============================================================"
echo -e "  ${GREEN}All services running!${NC}"
echo "============================================================"
echo ""
echo "  Gateway:       http://localhost:8000"
echo "  Console:       http://localhost:8000/console"
echo "  API Docs:      http://localhost:8000/docs"
echo ""
echo "  MCP Servers:"
echo "    Pathology:   http://localhost:8001"
echo "    Scoring:     http://localhost:8002"
echo "    Data Lake:   http://localhost:8003"
echo ""
echo "  API Key:       ok-demo-key"
echo ""
echo "  Press Ctrl+C to stop all services."
echo ""

# Wait for all background jobs
wait
