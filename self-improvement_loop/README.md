# 📈 AMZN Daily Close Predictor — Self-Improving Agentic Loop

A daily experiment: every trading day, **before the US market opens**, a Claude-powered ensemble
predicts Amazon's (AMZN) closing price. The next trading day it checks the real close, grades
itself **PASS/FAIL (±1%)**, writes a post-mortem, and refines its own strategy. State lives in the
repo — git history *is* the experiment log.

> Research experiment, not financial advice. It places no trades and moves no money.

## How it works (one run/day)
1. **Guard** — skip non-NYSE days.
2. **Score yesterday** — fetch the real close, compute PASS/FAIL, directional hit, error vs a
   random-walk baseline; update per-strategy scorecards; write a learning.
3. **Predict today** — 5 analyst lenses (technical, momentum, contrarian, news-via-web-search,
   macro) each forecast the close; a meta-judge weights them by their *tracked* accuracy.
4. **Report** — regenerate `RESULTS.md`, `results/results.csv`, `results/metrics.json`, and the
   GitHub Pages dashboard.
5. **Commit** — push state back (only this project's paths).

## The bar for success
PASS = predicted close within **±1%** of actual. The headline verdict is **edge vs the random-walk
baseline** (`predicted = yesterday's close`): rolling-20-day MAPE must beat it, or the dashboard
honestly says "no edge yet."

## Self-improvement
- **Scorecards** (`learnings/scorecards.json`) track each strategy's accuracy → the meta-judge
  up-weights what works.
- **STRATEGY.md** is an append-only living strategy the reflector adds one tested insight to per day.
- **learnings/YYYY-MM-DD.md** post-mortems feed back into the next prediction.
- **learnings/FAILURES.md** — concentrated, append-only log of every missed prediction (>1% error)
  with a root-cause analysis (which analyst dragged the blend, direction vs magnitude, what to change).
- **learnings/WHATS_NOT_WORKING.md** — a rolling self-diagnosis regenerated each scoring that looks
  *across all failures* for recurring patterns; it's fed to the meta-judge before the next shot.
- **Analyst-level feedback** — each analyst also sees its own scorecard and the failure review, with
  an explicit anti-groupthink instruction (the diagnosed root cause of most misses).
- **Enforced gates** (`src/evals/gates.py`) — the diagnosis's fixes applied in code, not just prompts:
  the predicted move shrinks toward the prior close when analyst directional agreement is low or the
  rolling edge vs baseline is negative, and confidence is replaced by a calibrated value (rolling
  pass rate ± an analyst-agreement nudge). Fired gates are recorded per row (`gates_applied`).

## Layout
```
spec/        constitution, spec, plan, tasks (built spec-first)
src/         config, utils, data/, evals/, agents/, features, loop/, report
data/        predictions.jsonl (the ledger / state)
results/     results.csv, metrics.json, site/ (Pages dashboard)
learnings/   STRATEGY.md, scorecards.json, daily post-mortems
tests/       pytest suite
```

## Run locally
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY (+ FINNHUB_API_KEY recommended)

# offline end-to-end (no API spend, uses stub analysts + keyless yfinance data):
python -m src.loop.run_daily --mode backfill          # seed history
python -m src.loop.run_daily --mode daily --dry-run   # score + predict + report

# with keys, a real prediction (no commit):
python -m src.loop.run_daily --mode daily --no-commit
```
Modes: `daily` (default), `score`, `predict`, `report`, `backfill`. Flags: `--date YYYY-MM-DD`,
`--dry-run`, `--no-commit`.

## Activation on GitHub (one-time)
1. **Secrets** (repo → Settings → Secrets → Actions): `ANTHROPIC_API_KEY`, and `ALPHAVANTAGE_API_KEY`
   (recommended — also serves daily history reliably from CI) and/or `FINNHUB_API_KEY` (live quote).
   With no market-data key the keyless yfinance fallback is used (may be rate-limited on CI IPs).
2. **Pages** (Settings → Pages → Source: **GitHub Actions**) for the live dashboard URL.
3. **Default branch** — the workflow `.github/workflows/amzn-predict.yml` must be on the repo's
   default branch for the schedule to fire. Use the **Run workflow** button (`workflow_dispatch`) to
   smoke-test from any branch first.

The job runs `0 13 * * 1-5` UTC — before the 09:30 ET open year-round.

## Dashboard
Live: GitHub Pages URL (after enabling Pages). Markdown: [RESULTS.md](RESULTS.md).

## Data sources
Daily history & past closes: Alpha Vantage `TIME_SERIES_DAILY` (keyed) → yfinance (keyless) →
Stooq w/ proof-of-work solver (best-effort). Live/pre-market quote: Finnhub → Alpha Vantage →
yfinance → derived from history. The keyless path keeps validation working with no API key.
