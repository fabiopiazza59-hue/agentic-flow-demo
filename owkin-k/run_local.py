#!/usr/bin/env python3
"""
Run MCP Gateway locally without Docker.

Starts all MCP servers and the gateway in separate processes.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Get the project root
PROJECT_ROOT = Path(__file__).parent

# Load .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))

processes = []


def start_process(name, command, port, cwd=None):
    """Start a subprocess and return the process object."""
    print(f"[Starting] {name} on port {port}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    process = subprocess.Popen(
        command,
        cwd=cwd or PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    processes.append((name, process, port))
    return process


def cleanup(signum=None, frame=None):
    """Clean up all processes."""
    print("\n[Cleanup] Stopping all processes...")
    for name, proc, port in processes:
        print(f"  Stopping {name}...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[Cleanup] Done.")
    sys.exit(0)


def wait_for_health(port, timeout=30):
    """Wait for a service to become healthy."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n" + "=" * 60)
    print("  MCP GATEWAY - Local Development Server")
    print("=" * 60 + "\n")

    # Start MCP servers
    start_process(
        "Pathology Engine",
        [sys.executable, "-m", "uvicorn", "mcp_servers.pathology.server:app", "--host", "0.0.0.0", "--port", "8001"],
        8001
    )

    start_process(
        "Scoring Service",
        [sys.executable, "-m", "uvicorn", "mcp_servers.scoring.server:app", "--host", "0.0.0.0", "--port", "8002"],
        8002
    )

    start_process(
        "Data Lake",
        [sys.executable, "-m", "uvicorn", "mcp_servers.datalake.server:app", "--host", "0.0.0.0", "--port", "8003"],
        8003
    )

    # Wait for MCP servers to be ready
    print("\n[Waiting] MCP servers starting...")
    time.sleep(2)

    for name, proc, port in processes:
        if wait_for_health(port, timeout=15):
            print(f"  [OK] {name} is healthy")
        else:
            print(f"  [WARN] {name} health check timeout")

    # Start Gateway
    start_process(
        "Gateway",
        [sys.executable, "-m", "uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        8000
    )

    # Wait for gateway
    time.sleep(2)
    if wait_for_health(8000, timeout=15):
        print(f"  [OK] Gateway is healthy")
    else:
        print(f"  [WARN] Gateway health check timeout")

    print("\n" + "=" * 60)
    print("  All services running!")
    print("=" * 60)
    print(f"""
  Gateway:       http://localhost:8000
  Console:       http://localhost:8000/console
  API Docs:      http://localhost:8000/docs

  MCP Servers:
    Pathology:   http://localhost:8001
    Scoring:     http://localhost:8002
    Data Lake:   http://localhost:8003

  API Key:       ok-demo-key

  Press Ctrl+C to stop all services.
""")

    # Keep running and print output
    try:
        while True:
            for name, proc, port in processes:
                if proc.poll() is not None:
                    print(f"[ERROR] {name} exited with code {proc.returncode}")
                    # Read any remaining output
                    output, _ = proc.communicate()
                    if output:
                        print(output.decode())
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
