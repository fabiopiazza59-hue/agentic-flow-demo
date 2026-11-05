# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (28 sessions, 839 name-forecasts; mean daily APE gain -0.005%, CS [-0.018%, +0.009%]).

**Pending (2026-11-05):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.005% | [-0.018%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.002% | [-0.011%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.066 | [-0.316, +0.185] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0014 | [-0.004, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.066% | [-0.101%, +0.234%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.071% | [-0.090%, +0.233%] | ≈ not distinguishable |

MAPE — LLM 1.52%, free baseline 1.51%, prior close 1.58%; LLM 80% interval coverage 77.36%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 23 | -0.006% | [-0.045%, +0.033%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 22 | -0.002% | [-0.044%, +0.039%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 28 | 1.19% | 1.18% | 1 |
| ADBE | 28 | 2.01% | 2.01% | 0 |
| AMD | 28 | 2.77% | 2.71% | 1 |
| AMZN | 28 | 1.85% | 1.85% | 0 |
| AVGO | 28 | 1.92% | 1.90% | 4 |
| BAC | 28 | 0.98% | 0.99% | 1 |
| CAT | 28 | 2.18% | 2.15% | 1 |
| COST | 28 | 1.02% | 1.01% | 1 |
| CRM | 28 | 1.56% | 1.56% | 0 |
| CVX | 28 | 1.23% | 1.25% | 1 |
| DIS | 28 | 1.06% | 1.07% | 1 |
| GOOGL | 28 | 1.22% | 1.24% | 1 |
| HD | 28 | 1.53% | 1.54% | 1 |
| JNJ | 28 | 1.59% | 1.57% | 1 |
| JPM | 28 | 0.94% | 0.93% | 1 |
| KO | 28 | 0.94% | 0.92% | 1 |
| LLY | 28 | 1.35% | 1.38% | 3 |
| MA | 28 | 1.29% | 1.29% | 1 |
| META | 27 | 1.90% | 1.90% | 0 |
| MSFT | 28 | 1.39% | 1.38% | 2 |
| NFLX | 28 | 1.59% | 1.59% | 0 |
| NVDA | 28 | 1.42% | 1.38% | 3 |
| ORCL | 28 | 3.33% | 3.27% | 2 |
| PEP | 28 | 1.26% | 1.26% | 0 |
| PG | 28 | 1.26% | 1.27% | 1 |
| TSLA | 28 | 1.57% | 1.57% | 0 |
| UNH | 28 | 1.42% | 1.41% | 1 |
| V | 28 | 1.17% | 1.17% | 0 |
| WMT | 28 | 1.34% | 1.34% | 1 |
| XOM | 28 | 1.31% | 1.33% | 1 |

_Research experiment, not financial advice._
