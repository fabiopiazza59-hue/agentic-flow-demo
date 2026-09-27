# Specification — AMZN Daily Close Predictor

## 1. Overview
A single daily process (`src/loop/run_daily.py`) runs before the US market open on each trading day.
It (A) scores the previous trading day's prediction against the real close, (B) reflects on the
result, (C) predicts today's close via a strategy ensemble + meta-judge, and (D) regenerates
experiment-tracking artifacts and commits them.

## 2. Definitions
- **Trading day (T):** an NYSE session day (per `pandas_market_calendars`).
- **Previous trading day (T-1):** the most recent NYSE session before T.
- **Actual close:** the official AMZN regular-session closing price for a given session.
- **Prediction:** a forecast of AMZN's close for session T, made before T's open.
- **Prior close:** actual close of T-1 (the baseline anchor and direction reference).

## 3. Eval definitions (the contract the loop optimizes)
For a scored prediction with `predicted_close` p and `actual_close` a (a from T's session):
- **Absolute percent error:** `ape = |p − a| / a`.
- **PASS (primary):** `ape ≤ 0.01` (within ±1%). Else **FAIL**.
- **Direction:** `up` if predicted/actual close > prior close, else `down` (flat → `down`).
  - **Directional hit:** predicted direction == actual direction.
- **Random-walk baseline:** `baseline_pred = prior_close`; `baseline_ape = |prior_close − a| / a`.
- **Beats baseline (per-day):** `ape < baseline_ape`.
- **Brier component:** with confidence c ∈ [0,1] and outcome o ∈ {1 PASS, 0 FAIL}: `(c − o)²`.

Aggregates (rolling window N = 20 scored days, and all-time):
- `pass_rate` = mean(PASS).
- `directional_accuracy` = mean(directional hit).
- `mape` = mean(ape); `baseline_mape` = mean(baseline_ape).
- `edge` = `baseline_mape − mape` (positive ⇒ real skill). **Headline verdict.**
- `brier` = mean(Brier component) (lower = better-calibrated confidence).

## 4. Data contracts
### 4.1 Normalized quote (`src/data/providers.py` → `get_quote(symbol)`)
```
Quote = {
  symbol: str,
  last: float | None,        # latest/most-recent price (pre-market or last trade)
  prev_close: float,         # previous session close (authoritative anchor)
  open: float | None,
  high: float | None,
  low: float | None,
  asof: str (ISO date),      # session date the prev_close refers to
  source: str                # "finnhub" | "alphavantage" | "stooq"
}
```
Quote provider order: Finnhub `/quote` (keyed) → Alpha Vantage `GLOBAL_QUOTE` (keyed) → yfinance
fast_info (keyless) → derived from the latest completed session in history.
`get_actual_close(symbol, date)` returns the official close for a specific past session via the
history chain below.

### 4.2 Daily history (`get_history(symbol, lookback_days)`)
Returns a pandas DataFrame indexed by date with columns `[open, high, low, close, volume]`, used by
`features.py`. History provider order: Alpha Vantage `TIME_SERIES_DAILY` (keyed, reliable from CI
IPs) → yfinance (keyless) → Stooq CSV with proof-of-work solver (keyless, best-effort last resort).
Note: Stooq now gates CSV behind a JS proof-of-work + API key, so yfinance is the practical keyless
source and Alpha Vantage the recommended keyed source for CI.

### 4.3 Ledger row (`data/predictions.jsonl`, one JSON object per line)
```
{
  "date": "YYYY-MM-DD",            # session T this prediction targets
  "created_at": "ISO-8601",
  "prior_close": float,
  "predicted_close": float,
  "predicted_direction": "up"|"down",
  "confidence": float,             # 0..1
  "weights": { "<strategy>": float, ... },  # meta-judge weights, sum≈1
  "analyst_predictions": { "<strategy>": {"predicted_close": float, "direction": "...",
                                           "confidence": float, "rationale": str}, ... },
  "rationale": str,                # meta-judge synthesis rationale
  "status": "pending" | "scored",
  // filled at scoring (T+1):
  "actual_close": float|null,
  "ape": float|null,
  "pass": bool|null,
  "directional_hit": bool|null,
  "baseline_ape": float|null,
  "beats_baseline": bool|null,
  "winning_strategy": str|null     # analyst closest to actual
}
```

### 4.4 Scorecards (`learnings/scorecards.json`)
```
{ "<strategy>": { "n": int, "wins": int, "sum_ape": float,
                  "hit_rate": float, "mape": float, "weight_hint": float }, ... }
```
`weight_hint` is derived from recent accuracy and fed to the meta-judge as a prior.

## 5. Functional requirements
- **FR1 Trading-day guard.** On a non-trading day, exit 0 with no writes.
- **FR2 Score (T-1).** Locate the pending ledger row for T-1; fetch its actual close; compute all
  eval fields; set `status="scored"`; update scorecards; write a post-mortem learning file.
  Idempotent: re-running a scored day does not double-count.
- **FR3 Reflect.** Reflector (Opus) writes `learnings/YYYY-MM-DD.md` (what happened, why, one
  concrete adjustment) and appends a bounded note to `learnings/STRATEGY.md` (never deletes). On a
  **FAIL** it also (a) appends a root-cause entry to `learnings/FAILURES.md` (concentrated, append-only
  log of every miss) and (b) regenerates `learnings/WHATS_NOT_WORKING.md` — a rolling self-diagnosis
  across ALL scored predictions surfacing recurring failure patterns. FR4's predict step feeds
  `WHATS_NOT_WORKING.md` to the meta-judge so the next prediction actively avoids known mistakes.
- **FR4 Predict (T).** Build features; run analysts in parallel; meta-judge synthesizes final
  prediction using scorecards + STRATEGY.md + last-N learnings; append a `pending` ledger row.
  Exactly one pending row per target date (re-running replaces same-day pending row).
- **FR5 Report.** Regenerate `results/results.csv`, `results/metrics.json`, `RESULTS.md`, and
  `results/site/{index.html,data.json}` from the ledger.
- **FR6 Commit.** Stage only `self-improvement_loop/{data,results,learnings,RESULTS.md,README.md}`;
  commit and push. No-op if nothing changed.
- **FR7 Modes.** `--mode daily` (default: score→reflect→predict→report), `score`, `predict`,
  `report`, `backfill` (seed history/baseline). `--dry-run` skips commits and uses mocks if no keys.
- **FR8 Resilience.** Keyed-provider failure falls back to Stooq. Malformed model JSON is repaired
  via `extract_json`; if an analyst fails, it is dropped and the meta-judge proceeds with the rest.

## 6. Non-functional requirements
- **NFR1 Cost:** ≤ ~$0.20/day API spend (5 Sonnet analysts + Opus judge + Opus reflector).
- **NFR2 Determinism of evals:** metric functions are pure and unit-tested at the ±1% boundary.
- **NFR3 Reproducibility:** all state in-repo; given the ledger, all results/* are regenerable via
  `--mode report`.
- **NFR4 Time correctness:** scheduled at 13:00 UTC → before 09:30 ET open year-round (08:00 EST /
  09:00 EDT). Trading-day guard handles holidays.
- **NFR5 Monorepo safety:** never stage paths outside `self-improvement_loop/`.

## 7. Out of scope
Order execution, portfolio/PNL accounting, options/derivatives, intraday prediction, multi-symbol
support (AMZN only for v1; symbol is configurable but untested for others).
