#!/usr/bin/env python3
"""
Scalp MVP - CLI Test Script

Simple command-line interface for testing the scalp trading assistant.

Usage:
    python cli.py "Analyze NVDA for scalp entry"
    python cli.py "Check TSLA setup, RSI 35, volume 1.2x"
    python cli.py  # Interactive mode
"""

import sys
import httpx
import json

API_URL = "http://localhost:8000"


def analyze(query: str) -> dict:
    """Send analysis request to the API."""
    try:
        response = httpx.post(
            f"{API_URL}/analyze",
            json={"query": query},
            timeout=60.0  # Longer timeout for LLM calls
        )
        response.raise_for_status()
        return response.json()
    except httpx.ConnectError:
        return {"error": "Cannot connect to server. Run: python main.py"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def check_health() -> bool:
    """Check if the server is running."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        return response.status_code == 200
    except:
        return False


def print_result(result: dict):
    """Pretty print the analysis result."""
    print("\n" + "=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    if result.get("success"):
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"Intent: {result.get('intent', 'N/A')}")
        print("-" * 60)
        print("\nRESPONSE:")
        print(result.get("response", "No response"))
    else:
        print(f"Failed: {json.dumps(result, indent=2)}")

    print("=" * 60 + "\n")


def interactive_mode():
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("  SCALP MVP - Interactive CLI")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    if not check_health():
        print("WARNING: Server not responding. Start it with: python main.py\n")

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            print("\nAnalyzing...")
            result = analyze(query)
            print_result(result)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command-line argument mode
        query = " ".join(sys.argv[1:])
        print(f"\nAnalyzing: {query}")
        result = analyze(query)
        print_result(result)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
