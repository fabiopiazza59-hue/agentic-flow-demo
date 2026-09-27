"""
CLI client: submit a query and follow the run's event stream.

    python cli.py "Price an Asian call on NVDA, spot 140, strike 150, vol 45%, 1 year"
    python cli.py --url http://localhost:8000 "What is the Sharpe ratio?"
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a query to the harness and stream events")
    parser.add_argument("query")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--callback-url", default=None, help="optional webhook the harness will POST the final result to")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=None) as client:
        resp = client.post("/runs", json={"query": args.query, "callback_url": args.callback_url})
        resp.raise_for_status()
        run = resp.json()
        print(f"run {run['run_id']} accepted (status={run['status']})\n")

        with client.stream("GET", run["events_url"]) as stream:
            event_name = None
            for line in stream.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip() or "{}")
                    if event_name == "end":
                        break
                    _print_event(event_name or data.get("type", "?"), data)
                elif not line:
                    event_name = None

        final = client.get(f"/runs/{run['run_id']}").json()
        print("\n" + "=" * 64)
        print(f"status: {final['status']}   resumes: {final['resume_count']}")
        for job in final["jobs"]:
            print(f"job {job['job_id']}: {job['kind']} -> {job['status']} via {job['delivered_via']} on {job['backend']}")
        print("=" * 64)
        print(final.get("output") or final.get("error") or "")
    return 0


def _print_event(name: str, data: dict) -> None:
    payload = {k: v for k, v in data.items() if k not in ("seq", "type", "ts", "output")}
    print(f"[{data.get('seq', '?'):>2}] {name:<20} {json.dumps(payload, default=str)[:160]}")


if __name__ == "__main__":
    sys.exit(main())
