# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (24 sessions, 719 name-forecasts; mean daily APE gain -0.006%, CS [-0.018%, +0.007%]).

**Pending (2026-10-30):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 3 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.006% | [-0.018%, +0.007%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.003% | [-0.011%, +0.006%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.093 | [-0.365, +0.179] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0018 | [-0.005, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.060% | [-0.112%, +0.231%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.065% | [-0.104%, +0.234%] | ≈ not distinguishable |

MAPE — LLM 1.39%, free baseline 1.39%, prior close 1.45%; LLM 80% interval coverage 79.01%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 19 | -0.010% | [-0.054%, +0.034%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 18 | +0.002% | [-0.028%, +0.033%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 24 | 0.93% | 0.93% | 0 |
| ADBE | 24 | 2.07% | 2.07% | 0 |
| AMD | 24 | 2.34% | 2.34% | 0 |
| AMZN | 24 | 1.49% | 1.49% | 0 |
| AVGO | 24 | 1.89% | 1.87% | 4 |
| BAC | 24 | 0.82% | 0.84% | 1 |
| CAT | 24 | 2.05% | 2.02% | 1 |
| COST | 24 | 0.86% | 0.85% | 1 |
| CRM | 24 | 1.59% | 1.59% | 0 |
| CVX | 24 | 1.06% | 1.06% | 0 |
| DIS | 24 | 0.94% | 0.95% | 1 |
| GOOGL | 24 | 1.22% | 1.24% | 1 |
| HD | 24 | 1.28% | 1.29% | 1 |
| JNJ | 24 | 1.48% | 1.47% | 1 |
| JPM | 24 | 0.75% | 0.74% | 1 |
| KO | 24 | 0.94% | 0.92% | 1 |
| LLY | 24 | 1.48% | 1.48% | 2 |
| MA | 24 | 1.13% | 1.14% | 1 |
| META | 23 | 1.97% | 1.97% | 0 |
| MSFT | 24 | 1.04% | 1.02% | 2 |
| NFLX | 24 | 1.54% | 1.54% | 0 |
| NVDA | 24 | 1.30% | 1.26% | 3 |
| ORCL | 24 | 3.07% | 3.01% | 2 |
| PEP | 24 | 1.08% | 1.08% | 0 |
| PG | 24 | 1.21% | 1.22% | 1 |
| TSLA | 24 | 1.64% | 1.64% | 0 |
| UNH | 24 | 1.38% | 1.36% | 1 |
| V | 24 | 1.03% | 1.03% | 0 |
| WMT | 24 | 1.17% | 1.18% | 1 |
| XOM | 24 | 1.06% | 1.06% | 0 |

_Research experiment, not financial advice._
