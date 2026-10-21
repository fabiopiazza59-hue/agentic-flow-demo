# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (17 sessions, 509 name-forecasts; mean daily APE gain -0.007%, CS [-0.025%, +0.011%]).

**Pending (2026-10-21):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.007% | [-0.025%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.004% | [-0.015%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.139 | n=7 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | -0.0014 | [-0.006, +0.003] | ≈ not distinguishable |
| APE gain: LLM vs prior close | -0.027% | [-0.120%, +0.066%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | -0.020% | [-0.108%, +0.068%] | ≈ not distinguishable |

MAPE — LLM 1.36%, free baseline 1.35%, prior close 1.33%; LLM 80% interval coverage 78.99%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 12 | +0.007% | [-0.038%, +0.052%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 11 | +0.010% | [-0.031%, +0.051%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 17 | 0.89% | 0.89% | 0 |
| ADBE | 17 | 1.87% | 1.87% | 0 |
| AMD | 17 | 2.20% | 2.20% | 0 |
| AMZN | 17 | 1.39% | 1.39% | 0 |
| AVGO | 17 | 1.99% | 1.96% | 4 |
| BAC | 17 | 0.98% | 1.00% | 1 |
| CAT | 17 | 1.55% | 1.55% | 0 |
| COST | 17 | 0.80% | 0.79% | 1 |
| CRM | 17 | 1.70% | 1.70% | 0 |
| CVX | 17 | 1.09% | 1.09% | 0 |
| DIS | 17 | 0.95% | 0.96% | 1 |
| GOOGL | 17 | 1.08% | 1.08% | 0 |
| HD | 17 | 1.19% | 1.20% | 1 |
| JNJ | 17 | 1.56% | 1.54% | 1 |
| JPM | 17 | 0.77% | 0.75% | 1 |
| KO | 17 | 0.83% | 0.83% | 0 |
| LLY | 17 | 1.42% | 1.42% | 0 |
| MA | 17 | 1.02% | 1.02% | 0 |
| META | 16 | 1.78% | 1.78% | 0 |
| MSFT | 17 | 0.98% | 0.97% | 1 |
| NFLX | 17 | 1.50% | 1.50% | 0 |
| NVDA | 17 | 1.32% | 1.26% | 3 |
| ORCL | 17 | 3.23% | 3.13% | 2 |
| PEP | 17 | 1.04% | 1.04% | 0 |
| PG | 17 | 1.13% | 1.13% | 0 |
| TSLA | 17 | 1.62% | 1.62% | 0 |
| UNH | 17 | 1.57% | 1.54% | 1 |
| V | 17 | 1.00% | 1.00% | 0 |
| WMT | 17 | 1.25% | 1.26% | 1 |
| XOM | 17 | 1.04% | 1.04% | 0 |

_Research experiment, not financial advice._
