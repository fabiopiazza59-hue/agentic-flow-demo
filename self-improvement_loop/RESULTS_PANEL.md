# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (3 sessions, 90 name-forecasts; mean daily APE gain 0.005%, CS n=3 sessions (too few)).

**Pending (2026-10-01):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.005% | n=3 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.002% | n=3 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0028 | n=3 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.037% | n=3 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.042% | n=3 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.28%, free baseline 1.28%, prior close 1.24%; LLM 80% interval coverage 75.56%; names moved per session 1.3. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 3 | 1.50% | 1.50% | 0 |
| ADBE | 3 | 1.82% | 1.82% | 0 |
| AMD | 3 | 1.57% | 1.57% | 0 |
| AMZN | 3 | 0.81% | 0.81% | 0 |
| AVGO | 3 | 1.50% | 1.50% | 0 |
| BAC | 3 | 1.47% | 1.47% | 0 |
| CAT | 3 | 1.03% | 1.03% | 0 |
| COST | 3 | 0.67% | 0.63% | 1 |
| CRM | 3 | 1.88% | 1.88% | 0 |
| CVX | 3 | 0.73% | 0.73% | 0 |
| DIS | 3 | 0.19% | 0.24% | 1 |
| GOOGL | 3 | 0.51% | 0.51% | 0 |
| HD | 3 | 1.08% | 1.16% | 1 |
| JNJ | 3 | 1.04% | 1.04% | 0 |
| JPM | 3 | 1.34% | 1.34% | 0 |
| KO | 3 | 0.67% | 0.67% | 0 |
| LLY | 3 | 0.95% | 0.95% | 0 |
| MA | 3 | 1.05% | 1.05% | 0 |
| META | 3 | 3.27% | 3.27% | 0 |
| MSFT | 3 | 0.82% | 0.82% | 0 |
| NFLX | 3 | 1.93% | 1.93% | 0 |
| NVDA | 3 | 0.90% | 0.90% | 0 |
| ORCL | 3 | 2.82% | 2.82% | 0 |
| PEP | 3 | 0.63% | 0.63% | 0 |
| PG | 3 | 1.63% | 1.63% | 0 |
| TSLA | 3 | 1.99% | 1.99% | 0 |
| UNH | 3 | 0.99% | 0.99% | 0 |
| V | 3 | 0.85% | 0.85% | 0 |
| WMT | 3 | 1.76% | 1.82% | 1 |
| XOM | 3 | 0.96% | 0.96% | 0 |

_Research experiment, not financial advice._
