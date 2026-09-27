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

    # --- integrity: a prediction only counts if it was made before the session opened ---
    SESSION_OPEN_UTC_HOUR: int = 13       # 09:30 ET regular-session open, in UTC
    SESSION_OPEN_UTC_MINUTE: int = 30
    ENFORCE_PRE_OPEN: bool = True         # refuse to write a prediction row after the open
    PREDICT_CUTOFF_MINUTES: int = 10      # a research run (LLM calls) must start this early
    REFRESH_CUTOFF_MINUTES: int = 3       # a code-only anchor refresh may run until this close
    SCORE_SETTLE_MINUTES: int = 30        # wait this long after the close before scoring

    # --- v2 pipeline (spec/v2-sota-upgrade.md) ---
    PIPELINE_VERSION: str = "v2"
    A_JUDGE_MAX_SIGMA: float = 0.5        # the judge may move the analyst blend at most ±0.5σ
    DEFAULT_SIGMA: float = 0.018          # daily σ fallback when no history is available
    LAB_ALPHA: float = 0.05               # anytime-valid error rate for lab decisions

    # --- models ---
    ANALYST_MODEL: str = "claude-sonnet-4-6"
    JUDGE_MODEL: str = "claude-opus-4-8"
    REFLECTOR_MODEL: str = "claude-opus-4-8"
    ANALYST_MAX_TOKENS: int = 1024
    JUDGE_MAX_TOKENS: int = 1600
    REFLECTOR_MAX_TOKENS: int = 1400
    LEARNINGS_CONTEXT_N: int = 5          # recent post-mortems fed into predict step

    # --- guardrail gates (src/evals/gates.py) ---
    GATE_EDGE_WINDOW: int = 10            # scored days used for the rolling-edge gate
    GATE_MIN_SCORED: int = 5              # min real scored days before edge/confidence gating
    GATE_CONSENSUS_MIN: float = 0.6       # below this analyst agreement, shrink the move
    GATE_SHRINK: float = 0.5              # move multiplier per fired gate

    # --- variant B: raven-style prior + pulse + bounded decider (src/variant_b/) ---
    B_PULSE_MODEL: str = "claude-sonnet-5"     # evidence gathering (web search)
    B_DECIDER_MODEL: str = "claude-opus-4-8"   # flagship bounded decision
    B_PULSE_MAX_TOKENS: int = 4096
    B_DECIDER_MAX_TOKENS: int = 1200
    B_DRIFT_WINDOW: int = 60              # sessions for the drift estimate
    B_SIGMA_WINDOW: int = 20              # sessions for the daily-vol estimate
    B_MC_PATHS: int = 10000               # Monte-Carlo draws for prior percentiles
    B_MAX_ADJ_SIGMA: float = 0.8          # evidence may move the prior at most ±0.8σ
    B_MAX_MOVE_SIGMA: float = 1.5         # hard cap on total move from prev close, in σ
    B_FAILURES_TAIL_CHARS: int = 3000     # tail of B's failure log fed to the decider
    AB_MIN_PAIRED_DAYS: int = 10          # below this, the A/B report renders no verdict

    # --- panel: many names per session, for statistical power (src/panel/, spec/panel.md) ---
    PANEL_SYMBOLS: tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "CRM",
        "ADBE", "AMD", "NFLX", "JPM", "BAC", "V", "MA", "UNH", "JNJ", "LLY",
        "XOM", "CVX", "WMT", "COST", "PG", "KO", "PEP", "HD", "DIS", "CAT",
    )
    PANEL_MODEL: str = "claude-sonnet-5"       # same model as arm B's evidence pulse
    PANEL_MAX_TOKENS: int = 8000
    PANEL_BATCH_SIZE: int = 10                 # names per LLM call
    PANEL_WEB_SEARCHES: int = 3                # per call
    PANEL_MAX_ADJ_SIGMA: float = 0.5           # the LLM may move a name off its anchor ±0.5σ
    PANEL_MIN_NAMES: int = 10                  # fewer usable names -> wait for the next run

    # --- prompt evolution on the panel (src/panel/evolve.py) ---
    EVOLVE_MAX_CHALLENGERS: int = 2            # shadow prompt variants per session
    EVOLVE_MIN_SESSIONS: int = 5               # champion history needed before proposing
    EVOLVE_MAX_SESSIONS: int = 60              # retire a challenger still undecided after this
    EVOLVE_MAX_STRATEGY_CHARS: int = 3000

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
    LEDGER_B_PATH: Path = PROJECT_ROOT / "data" / "predictions_b.jsonl"
    LEARNINGS_B_DIR: Path = PROJECT_ROOT / "learnings_b"
    AB_COMPARE_JSON: Path = PROJECT_ROOT / "results" / "ab_compare.json"
    PANEL_LEDGER_PATH: Path = PROJECT_ROOT / "data" / "panel.jsonl"
    PANEL_STATE_PATH: Path = PROJECT_ROOT / "learnings" / "panel_state.json"
    PANEL_METRICS_JSON: Path = PROJECT_ROOT / "results" / "panel_metrics.json"
    PANEL_RESULTS_MD: Path = PROJECT_ROOT / "RESULTS_PANEL.md"
    PROMPT_REGISTRY_PATH: Path = PROJECT_ROOT / "learnings" / "prompt_variants.json"

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
        for d in (self.DATA_DIR, self.RESULTS_DIR, self.SITE_DIR, self.LEARNINGS_DIR,
                  self.LEARNINGS_B_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
