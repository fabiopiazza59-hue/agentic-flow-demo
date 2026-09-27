#!/usr/bin/env python3
"""CLI entry point: run the full or partial Seamaster pipeline.

Usage:
    # Full pipeline (Stage 1-6)
    python run_pipeline.py

    # Start from Stage 3 with pre-built Stage 2 output
    python run_pipeline.py --start-stage 3 --stage2-input fixtures/sample_stage2_output.yaml

    # Provide current prices for Stage 4a
    python run_pipeline.py --start-stage 3 --stage2-input fixtures/sample_stage2_output.yaml \\
        --prices "MU=268.50,KRE=48.20,CCJ=55.00,OKLO=28.00"

    # Custom account size
    python run_pipeline.py --capital 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.settings import Settings, AccountConfig, PipelineConfig
from src.models.schemas import AccountState
from src.pipeline import run_pipeline


def parse_prices(prices_str: str) -> dict[str, float]:
    """Parse 'MU=268.50,KRE=48.20' into dict."""
    if not prices_str:
        return {}
    result = {}
    for pair in prices_str.split(","):
        ticker, price = pair.strip().split("=")
        result[ticker.strip()] = float(price.strip())
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the Seamaster trading advisor pipeline"
    )
    parser.add_argument(
        "--start-stage", type=int, default=1, choices=range(1, 7),
        help="Stage to start from (default: 1)"
    )
    parser.add_argument(
        "--stage2-input", type=str, default=None,
        help="Path to Stage 2 output YAML (required if --start-stage >= 3)"
    )
    parser.add_argument(
        "--prices", type=str, default=None,
        help="Current prices as 'TICKER=PRICE,...' (for Stage 4a)"
    )
    parser.add_argument(
        "--capital", type=float, default=2000,
        help="Account capital in USD (default: 2000)"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Override run ID"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override Claude model (e.g., claude-sonnet-4-20250514)"
    )
    args = parser.parse_args()

    # Build settings
    account = AccountConfig(total_capital_usd=args.capital)
    pipeline_cfg = PipelineConfig()
    if args.model:
        pipeline_cfg.model = args.model
    settings = Settings(account=account, pipeline=pipeline_cfg)

    # Build account state
    account_state = AccountState(
        total_capital_usd=args.capital,
        available_capital_usd=args.capital,
        current_portfolio_heat_pct=0.0,
    )

    # Parse prices
    prices = parse_prices(args.prices) if args.prices else {}

    result = run_pipeline(
        settings=settings,
        account_state=account_state,
        start_stage=args.start_stage,
        stage2_input=args.stage2_input,
        current_prices=prices,
        run_id=args.run_id,
    )

    # Print the action table if Stage 6 ran
    if result.stage6:
        from src.stages.stage6_action_table import render_markdown
        print(render_markdown(result.stage6))


if __name__ == "__main__":
    main()
