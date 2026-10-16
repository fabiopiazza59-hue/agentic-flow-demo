# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (15 sessions, 449 name-forecasts; mean daily APE gain -0.004%, CS [-0.017%, +0.010%]).

**Pending (2026-10-19):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 3 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.004% | [-0.017%, +0.010%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.002% | [-0.014%, +0.010%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.112 | n=6 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0011 | [-0.005, +0.003] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.019% | [-0.123%, +0.085%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.015% | [-0.117%, +0.086%] | ≈ not distinguishable |

MAPE — LLM 1.38%, free baseline 1.38%, prior close 1.37%; LLM 80% interval coverage 77.97%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 10 | -0.002% | [-0.040%, +0.036%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 9 | +0.004% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 15 | 0.93% | 0.93% | 0 |
| ADBE | 15 | 1.76% | 1.76% | 0 |
| AMD | 15 | 2.31% | 2.31% | 0 |
| AMZN | 15 | 1.22% | 1.22% | 0 |
| AVGO | 15 | 2.10% | 2.11% | 3 |
| BAC | 15 | 0.97% | 1.00% | 1 |
| CAT | 15 | 1.70% | 1.70% | 0 |
| COST | 15 | 0.84% | 0.83% | 1 |
| CRM | 15 | 1.84% | 1.84% | 0 |
| CVX | 15 | 1.19% | 1.19% | 0 |
| DIS | 15 | 0.90% | 0.90% | 1 |
| GOOGL | 15 | 1.04% | 1.04% | 0 |
| HD | 15 | 1.22% | 1.23% | 1 |
| JNJ | 15 | 1.60% | 1.57% | 1 |
| JPM | 15 | 0.73% | 0.71% | 1 |
| KO | 15 | 0.91% | 0.91% | 0 |
| LLY | 15 | 1.30% | 1.30% | 0 |
| MA | 15 | 1.11% | 1.11% | 0 |
| META | 14 | 1.92% | 1.92% | 0 |
| MSFT | 15 | 1.02% | 1.01% | 1 |
| NFLX | 15 | 1.55% | 1.55% | 0 |
| NVDA | 15 | 1.33% | 1.30% | 2 |
| ORCL | 15 | 3.42% | 3.36% | 1 |
| PEP | 15 | 1.07% | 1.07% | 0 |
| PG | 15 | 1.14% | 1.14% | 0 |
| TSLA | 15 | 1.47% | 1.47% | 0 |
| UNH | 15 | 1.48% | 1.46% | 1 |
| V | 15 | 1.05% | 1.05% | 0 |
| WMT | 15 | 1.35% | 1.36% | 1 |
| XOM | 15 | 1.12% | 1.12% | 0 |

_Research experiment, not financial advice._
