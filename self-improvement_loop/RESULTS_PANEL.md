# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (26 sessions, 779 name-forecasts; mean daily APE gain -0.003%, CS [-0.017%, +0.010%]).

**Pending (2026-11-03):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 1 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.003% | [-0.017%, +0.010%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.010%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.048 | [-0.346, +0.250] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0015 | [-0.004, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.079% | [-0.098%, +0.255%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.082% | [-0.089%, +0.253%] | ≈ not distinguishable |

MAPE — LLM 1.43%, free baseline 1.43%, prior close 1.51%; LLM 80% interval coverage 77.93%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 21 | -0.011% | [-0.051%, +0.028%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 20 | -0.006% | [-0.050%, +0.037%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 26 | 1.02% | 1.02% | 0 |
| ADBE | 26 | 1.97% | 1.97% | 0 |
| AMD | 26 | 2.68% | 2.68% | 0 |
| AMZN | 26 | 1.68% | 1.68% | 0 |
| AVGO | 26 | 1.91% | 1.89% | 4 |
| BAC | 26 | 0.89% | 0.91% | 1 |
| CAT | 26 | 2.09% | 2.06% | 1 |
| COST | 26 | 0.91% | 0.90% | 1 |
| CRM | 26 | 1.52% | 1.52% | 0 |
| CVX | 26 | 1.12% | 1.13% | 1 |
| DIS | 26 | 0.96% | 0.97% | 1 |
| GOOGL | 26 | 1.17% | 1.19% | 1 |
| HD | 26 | 1.32% | 1.33% | 1 |
| JNJ | 26 | 1.49% | 1.48% | 1 |
| JPM | 26 | 0.82% | 0.81% | 1 |
| KO | 26 | 0.93% | 0.91% | 1 |
| LLY | 26 | 1.44% | 1.47% | 3 |
| MA | 26 | 1.16% | 1.17% | 1 |
| META | 25 | 1.90% | 1.90% | 0 |
| MSFT | 26 | 1.13% | 1.12% | 2 |
| NFLX | 26 | 1.46% | 1.46% | 0 |
| NVDA | 26 | 1.36% | 1.32% | 3 |
| ORCL | 26 | 3.34% | 3.28% | 2 |
| PEP | 26 | 1.12% | 1.12% | 0 |
| PG | 26 | 1.20% | 1.21% | 1 |
| TSLA | 26 | 1.57% | 1.57% | 0 |
| UNH | 26 | 1.38% | 1.37% | 1 |
| V | 26 | 1.08% | 1.08% | 0 |
| WMT | 26 | 1.21% | 1.21% | 1 |
| XOM | 26 | 1.17% | 1.18% | 1 |

_Research experiment, not financial advice._
