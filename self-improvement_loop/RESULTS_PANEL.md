# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (30 sessions, 899 name-forecasts; mean daily APE gain -0.003%, CS [-0.017%, +0.011%]).

**Pending (2026-11-09):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.017%, +0.011%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.000% | [-0.011%, +0.010%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.041 | [-0.288, +0.207] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0013 | [-0.004, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.081% | [-0.079%, +0.241%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.083% | [-0.071%, +0.238%] | ≈ not distinguishable |

MAPE — LLM 1.58%, free baseline 1.58%, prior close 1.66%; LLM 80% interval coverage 78.32%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 25 | -0.010% | [-0.047%, +0.027%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 24 | -0.004% | [-0.042%, +0.034%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 30 | 1.18% | 1.17% | 1 |
| ADBE | 30 | 1.95% | 1.95% | 0 |
| AMD | 30 | 3.42% | 3.43% | 2 |
| AMZN | 30 | 1.81% | 1.81% | 0 |
| AVGO | 30 | 2.20% | 2.18% | 4 |
| BAC | 30 | 1.08% | 1.09% | 1 |
| CAT | 30 | 2.26% | 2.23% | 1 |
| COST | 30 | 1.00% | 0.99% | 1 |
| CRM | 30 | 1.59% | 1.59% | 0 |
| CVX | 30 | 1.22% | 1.23% | 1 |
| DIS | 30 | 1.06% | 1.06% | 1 |
| GOOGL | 30 | 1.26% | 1.28% | 1 |
| HD | 30 | 1.52% | 1.52% | 1 |
| JNJ | 30 | 1.59% | 1.58% | 1 |
| JPM | 30 | 1.06% | 1.05% | 1 |
| KO | 30 | 0.91% | 0.89% | 1 |
| LLY | 30 | 1.43% | 1.45% | 3 |
| MA | 30 | 1.29% | 1.30% | 1 |
| META | 29 | 1.94% | 1.94% | 0 |
| MSFT | 30 | 1.43% | 1.41% | 2 |
| NFLX | 30 | 1.60% | 1.60% | 0 |
| NVDA | 30 | 1.71% | 1.68% | 3 |
| ORCL | 30 | 3.36% | 3.31% | 2 |
| PEP | 30 | 1.18% | 1.18% | 0 |
| PG | 30 | 1.21% | 1.22% | 1 |
| TSLA | 30 | 1.95% | 1.95% | 0 |
| UNH | 30 | 1.47% | 1.46% | 1 |
| V | 30 | 1.21% | 1.21% | 0 |
| WMT | 30 | 1.34% | 1.34% | 1 |
| XOM | 30 | 1.31% | 1.32% | 1 |

_Research experiment, not financial advice._
