"""Central configuration for the AMZN daily close predictor.

Paths resolve relative to the project root so the code works the same locally and in CI.
Provider selection is driven by which API key is present in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # --- target & evals ---
    SYMBOL: str = "AMZN"
    PASS_THRESHOLD: float = 0.01          # ±1% absolute percent error => PASS
    ROLLING_WINDOW: int = 20              # trading days for rolling aggregates
    LOOKBACK_DAYS: int = 120             # history pulled for feature engineering

    # --- models ---
    ANALYST_MODEL: str = "claude-sonnet-4-6"
    JUDGE_MODEL: str = "claude-opus-4-8"
    REFLECTOR_MODEL: str = "claude-opus-4-8"
    ANALYST_MAX_TOKENS: int = 1024
    JUDGE_MAX_TOKENS: int = 1600
    REFLECTOR_MAX_TOKENS: int = 1400
    LEARNINGS_CONTEXT_N: int = 5          # recent post-mortems fed into predict step

    # --- paths ---
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"
    SITE_DIR: Path = PROJECT_ROOT / "results" / "site"
    LEARNINGS_DIR: Path = PROJECT_ROOT / "learnings"
    LEDGER_PATH: Path = PROJECT_ROOT / "data" / "predictions.jsonl"
    SCORECARDS_PATH: Path = PROJECT_ROOT / "learnings" / "scorecards.json"
    STRATEGY_PATH: Path = PROJECT_ROOT / "learnings" / "STRATEGY.md"
    RESULTS_CSV: Path = PROJECT_ROOT / "results" / "results.csv"
    METRICS_JSON: Path = PROJECT_ROOT / "results" / "metrics.json"
    RESULTS_MD: Path = PROJECT_ROOT / "RESULTS.md"
    SITE_DATA: Path = PROJECT_ROOT / "results" / "site" / "data.json"

    # --- env-derived helpers (not model fields) ---
    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def finnhub_api_key(self) -> str | None:
        return os.getenv("FINNHUB_API_KEY")

    @property
    def alphavantage_api_key(self) -> str | None:
        return os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")

    @property
    def keyed_provider(self) -> str | None:
        """Which keyed market-data provider to prefer, if any key is set."""
        if self.finnhub_api_key:
            return "finnhub"
        if self.alphavantage_api_key:
            return "alphavantage"
        return None

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.RESULTS_DIR, self.SITE_DIR, self.LEARNINGS_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
