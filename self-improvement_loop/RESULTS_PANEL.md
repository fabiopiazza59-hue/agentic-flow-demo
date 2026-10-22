# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (18 sessions, 539 name-forecasts; mean daily APE gain -0.008%, CS [-0.024%, +0.009%]).

**Pending (2026-10-22):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.008% | [-0.024%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.004% | [-0.015%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.156 | n=8 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0018 | [-0.006, +0.002] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.007% | [-0.119%, +0.105%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.001% | [-0.109%, +0.111%] | ≈ not distinguishable |

MAPE — LLM 1.32%, free baseline 1.31%, prior close 1.31%; LLM 80% interval coverage 79.79%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 13 | +0.007% | [-0.034%, +0.048%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 12 | +0.008% | [-0.029%, +0.046%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 18 | 0.94% | 0.94% | 0 |
| ADBE | 18 | 1.79% | 1.79% | 0 |
| AMD | 18 | 2.10% | 2.10% | 0 |
| AMZN | 18 | 1.34% | 1.34% | 0 |
| AVGO | 18 | 1.92% | 1.89% | 4 |
| BAC | 18 | 0.93% | 0.95% | 1 |
| CAT | 18 | 1.51% | 1.51% | 0 |
| COST | 18 | 0.80% | 0.80% | 1 |
| CRM | 18 | 1.62% | 1.62% | 0 |
| CVX | 18 | 1.07% | 1.07% | 0 |
| DIS | 18 | 0.92% | 0.93% | 1 |
| GOOGL | 18 | 1.12% | 1.12% | 0 |
| HD | 18 | 1.12% | 1.14% | 1 |
| JNJ | 18 | 1.49% | 1.47% | 1 |
| JPM | 18 | 0.73% | 0.71% | 1 |
| KO | 18 | 0.78% | 0.78% | 0 |
| LLY | 18 | 1.43% | 1.40% | 1 |
| MA | 18 | 0.99% | 0.99% | 0 |
| META | 17 | 1.70% | 1.70% | 0 |
| MSFT | 18 | 0.97% | 0.96% | 1 |
| NFLX | 18 | 1.44% | 1.44% | 0 |
| NVDA | 18 | 1.27% | 1.21% | 3 |
| ORCL | 18 | 3.22% | 3.13% | 2 |
| PEP | 18 | 1.05% | 1.05% | 0 |
| PG | 18 | 1.13% | 1.13% | 0 |
| TSLA | 18 | 1.62% | 1.62% | 0 |
| UNH | 18 | 1.48% | 1.46% | 1 |
| V | 18 | 0.97% | 0.97% | 0 |
| WMT | 18 | 1.22% | 1.23% | 1 |
| XOM | 18 | 1.00% | 1.00% | 0 |

_Research experiment, not financial advice._
