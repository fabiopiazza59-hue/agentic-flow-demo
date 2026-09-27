# 📈 AMZN Daily Close Predictor — Self-Improving Agentic Loop

A daily experiment: every trading day, **before the US market opens**, a Claude-powered ensemble
predicts Amazon's (AMZN) closing price as a calibrated distribution. After the close it scores
itself against the free baselines (the prior close, and the latest pre-open trade), writes a
post-mortem, and lets a measurement lab — not prose — decide which of its own mechanisms to keep.
State lives in the repo — git history *is* the experiment log.

**v2 (2026-09-27):** see [spec/v2-sota-upgrade.md](spec/v2-sota-upgrade.md) — what the record
actually showed (most rows were written after the open; the "pre-market quote" was the previous
close), and the timing model, proper scoring, anytime-valid verdicts and lab that fix it.

> Research experiment, not financial advice. It places no trades and moves no money.

## How it works (every scheduled run is idempotent)
1. **Score** — every pending forecast whose session has closed (+30 min to settle): PASS/FAIL,
   direction, error vs the random walk *and* vs the anchor; scorecards; a post-mortem.
2. **Pick the session** — the next session whose forecast window is open (previous close →
   10 min before its open). During a session there is nothing to forecast.
3. **Research it (once)** — 5 analyst lenses (technical, momentum, contrarian, news-via-web-search,
   macro) → the lab's champion blend → the judge's σ-sized adjustment (clamped in code) →
   guardrail gates → a calibrated predictive distribution. Arm B runs on the same snapshot.
4. **…or re-anchor it** — if it is already researched and still pre-open, move it onto the latest
   after-hours / pre-market trade in code (no LLM cost).
5. **Report + commit** — `RESULTS.md`, `results/*`, the Pages dashboard, `learnings/lab_state.json`.

The evening run after the close does the research hours before the bell, because GitHub starts
scheduled runs late (3.5–5.5 h in September 2026). Morning runs only re-anchor.

## The bar for success
The headline verdict is **edge vs the free baseline** — the latest pre-open trade the forecast was
anchored to (the prior close when there was none): all-time MAPE over **pre-open forecasts only**
must beat it, judged by an **anytime-valid 95% confidence sequence** on the paired daily gains, or
the dashboard honestly says "no edge yet." The CS stays valid although the page is re-read daily
(a daily fixed-sample test would "find" an edge 40% of the time under the null within a year).
CRPS, interval coverage and the Brier score of P(up) are reported alongside; PASS (±1%) and the
rolling-20 window are shown but mostly measure the day's volatility, so they don't drive it.

## Self-improvement
- **The lab** (`src/evals/lab.py`) — six rules for combining the analysts are replayed over every
  pre-open day with no lookahead; one replaces the equal-weight mean only when its anytime-valid
  CS (Bonferroni-corrected) proves it better. The same test switches the judge or the gates **off**
  once they are shown to hurt. Its state (`learnings/lab_state.json`) drives the next forecast.
- **Scorecards** (`learnings/scorecards.json`) track each strategy's accuracy (context for the
  judge and the analysts; the lab, not the scorecards, sets the blend).
- **STRATEGY.md** is an append-only living strategy the reflector adds one tested insight to per day.
- **learnings/YYYY-MM-DD.md** post-mortems feed back into the next prediction.
- **learnings/FAILURES.md** — concentrated, append-only log of every missed prediction (>1% error)
  with a root-cause analysis (which analyst dragged the blend, direction vs magnitude, what to change).
- **learnings/WHATS_NOT_WORKING.md** — a rolling self-diagnosis regenerated each scoring that looks
  *across all failures* for recurring patterns; it's fed to the meta-judge before the next shot.
- **Analyst-level feedback** — each analyst also sees its own scorecard and the failure review, with
  an explicit anti-groupthink instruction (the diagnosed root cause of most misses).
- **Pre-open integrity** (`src/evals/integrity.py`) — a row only counts as a forecast if it was
  created before its session opened. Every row carries `late_minutes`; `RESULTS.md` reports the
  record with and without the late ones, and the verdict is read off the pre-open column only.
  New post-open predictions are refused (`--allow-late` overrides, and still flags the row).
- **Verdict needs evidence** (`src/evals/paired.py`) — "edge confirmed" requires a paired
  day-level test on pre-open rows (90% bootstrap CI excluding zero *and* a sign test at p<0.05),
  so a nominal +0.02% reads as noise instead of a win.
- **Measurable gates** — each row stores `predicted_close_raw`, the pre-gate prediction, so the
  gates' own effect is reported as a paired per-day number instead of being assumed.
- **Enforced gates** (`src/evals/gates.py`) — the diagnosis's fixes applied in code, not just prompts:
  the predicted move shrinks toward the anchor when analyst directional agreement is low or the
  rolling pre-open edge vs the free baseline is negative. Fired gates are recorded per row
  (`gates_applied`); `confidence` is P(PASS) under the row's calibrated distribution.

## A/B test — two agent architectures, one daily run
Since 2026-07-18 every daily run executes **two arms on the same pre-open snapshot** (same
history, same quote — a fair paired test) and scores both the next day:

- **Arm A — ensemble + gates** (the original loop): 5 analyst lenses → meta-judge → deterministic
  guardrail gates. State: `data/predictions.jsonl`, `learnings/`.
- **Arm B — raven-style prior + pulse** (inspired by
  [predict-raven](https://github.com/Alchemist-X/predict-raven)): a statistical prior
  (drift/vol Monte-Carlo, centered on the live pre-open trade when there is one) → one
  evidence-gathering "pulse" agent (web search, audited to `learnings_b/pulse/`) → a decision
  agent whose adjustment is **hard-capped in code** at ±0.8σ (and the move from the prior center
  at 1.5σ). No ensemble, no meta-judge; failures feed straight back into the decider via
  `learnings_b/FAILURES.md`. State: `data/predictions_b.jsonl`, `learnings_b/`.

`results/ab_compare.json` + an A/B section in `RESULTS.md` and the dashboard track the paired
comparison on days where **both** forecasts were written before the open (APE and CRPS deltas,
anytime-valid CS; the sign test is descriptive). No verdict before 10 clean paired days.
Entry point: `python -m src.loop.run_ab` (CI uses this; `run_daily` still runs arm A standalone).

## Layout
```
spec/        constitution, spec, plan, tasks (built spec-first)
src/         config, utils, data/, evals/, agents/, features, loop/, report, report_ab
src/variant_b/  arm B: prior, pulse, decider, runner
data/        predictions.jsonl (arm A ledger), predictions_b.jsonl (arm B ledger)
results/     results.csv, metrics.json, ab_compare.json, site/ (Pages dashboard)
learnings/   STRATEGY.md, scorecards.json, lab_state.json, daily post-mortems (arm A)
learnings_b/ FAILURES.md, pulse/ audit artifacts (arm B)
tests/       pytest suite
```

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY (+ FINNHUB_API_KEY recommended)

# offline end-to-end (no API spend, uses stub analysts + keyless yfinance data).
# Run it on a copy of the project: offline stub forecasts would otherwise enter the real ledger.
python -m src.loop.run_daily --mode backfill          # seed history
python -m src.loop.run_ab --mode daily --dry-run      # score + research/re-anchor + report

# with keys, a real prediction (no commit):
python -m src.loop.run_daily --mode daily --no-commit
```
Modes: `daily` (default), `score`, `predict`, `report`, `backfill`. Flags: `--date YYYY-MM-DD`
(a past date is a replay, flagged late), `--dry-run`, `--no-commit`, `--allow-late`.

```bash
python -m tools.audit_review        # reproduce the findings in spec/improvements-from-2609.05663.md
python -m tools.backfill_integrity  # one-time: stamp late_minutes on historical rows
```

## Activation on GitHub (one-time)
1. **Secrets** (repo → Settings → Secrets → Actions): `ANTHROPIC_API_KEY`, and `ALPHAVANTAGE_API_KEY`
   (recommended — also serves daily history reliably from CI) and/or `FINNHUB_API_KEY` (live quote).
   With no market-data key the keyless yfinance fallback is used (may be rate-limited on CI IPs).
2. **Pages** (Settings → Pages → Source: **GitHub Actions**) for the live dashboard URL.
3. **Default branch** — the workflow `.github/workflows/amzn-predict.yml` must be on the repo's
   default branch for the schedule to fire, and it commits back to the branch it runs on. Use the
   **Run workflow** button (`workflow_dispatch`) to smoke-test from any branch first.

Schedule (UTC, weekdays): 21:41 and 01:17 (evening research), 09:07, 11:07, 12:37, 13:13 (morning
re-anchoring). Every slot is safe to run any time: it does only what the clock allows.

## Dashboard
Live: GitHub Pages URL (after enabling Pages). Markdown: [RESULTS.md](RESULTS.md).

## Data sources
Daily history & past closes: Alpha Vantage `TIME_SERIES_DAILY` (keyed) → yfinance (keyless) →
Stooq w/ proof-of-work solver (best-effort); a forecast waits until one of them has the previous
session's close. Pre-open anchor: Yahoo 1-minute bars with pre/post-market (keyless) → Finnhub
(keyed, timestamped) → the prior close. Alpha Vantage's free quote is not used as an anchor: before
the open it returns the previous close.
