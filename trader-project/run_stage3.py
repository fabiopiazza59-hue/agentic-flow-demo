#!/usr/bin/env python3
"""CLI entry point: run Stage 3 triage against a Stage 2 output file.

Usage:
    python run_stage3.py fixtures/sample_stage2_output.yaml
    python run_stage3.py fixtures/sample_stage2_output.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import Settings
from src.journal import write_journal
from src.stages.stage3_triage import load_stage2_from_yaml, run_stage3


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 triage gate")
    parser.add_argument("input", help="Path to Stage 2 output YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    parser.add_argument("--run-id", default=None, help="Override run ID")
    args = parser.parse_args()

    stage2 = load_stage2_from_yaml(args.input)
    settings = Settings()

    if args.dry_run:
        from src.stages.stage3_triage import _format_scenarios_for_prompt, SYSTEM_PROMPT
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER MESSAGE ===")
        print(_format_scenarios_for_prompt(stage2.scenarios))
        print(f"\nScenarios: {len(stage2.scenarios)}")
        return

    print(f"Running Stage 3 triage on {len(stage2.scenarios)} scenarios...")
    result = run_stage3(stage2, settings, run_id=args.run_id)

    # Write to journal
    journal_path = write_journal(3, result.batch_summary.batch_id, result)
    print(f"Journal written: {journal_path}")

    # Print summary
    summary = result.batch_summary
    print(f"\n{'='*60}")
    print(f"Batch: {summary.batch_id} | Scenarios: {summary.scenarios_in}")
    print(f"Verdicts: LIVE={summary.verdicts.get('LIVE', 0)} | "
          f"WATCH={summary.verdicts.get('WATCH', 0)} | "
          f"KILL={summary.verdicts.get('KILL', 0)}")
    print(f"Top 3: {', '.join(summary.top_3_by_conviction)}")
    print(f"Desk note: {summary.desk_note}")

    print(f"\n{'='*60}")
    for v in result.verdicts:
        icon = {"LIVE": "+", "WATCH": "~", "KILL": "x"}[v.verdict.value]
        print(f"[{icon}] {v.scenario_id} → {v.verdict.value}: {v.one_line_reason}")


if __name__ == "__main__":
    main()
