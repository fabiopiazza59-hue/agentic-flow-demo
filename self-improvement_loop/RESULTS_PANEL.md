# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (38 sessions, 1138 name-forecasts; mean daily APE gain -0.002%, CS [-0.013%, +0.009%]).

**Pending (2026-11-19):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.013%, +0.009%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.010%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0012 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.089% | [-0.042%, +0.220%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.091% | [-0.035%, +0.217%] | ≈ not distinguishable |

MAPE — LLM 1.59%, free baseline 1.59%, prior close 1.68%; LLM 80% interval coverage 78.64%; names moved per session 1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 33 | -0.011% | [-0.041%, +0.019%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 32 | -0.003% | [-0.032%, +0.026%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 38 | 1.11% | 1.11% | 1 |
| ADBE | 38 | 1.90% | 1.90% | 0 |
| AMD | 38 | 3.56% | 3.58% | 3 |
| AMZN | 38 | 1.77% | 1.77% | 0 |
| AVGO | 38 | 2.43% | 2.44% | 5 |
| BAC | 38 | 1.05% | 1.06% | 1 |
| CAT | 38 | 2.39% | 2.37% | 1 |
| COST | 38 | 0.93% | 0.92% | 1 |
| CRM | 38 | 1.71% | 1.71% | 0 |
| CVX | 38 | 1.29% | 1.30% | 1 |
| DIS | 38 | 1.08% | 1.08% | 2 |
| GOOGL | 38 | 1.29% | 1.31% | 2 |
| HD | 38 | 1.35% | 1.35% | 1 |
| JNJ | 38 | 1.49% | 1.48% | 1 |
| JPM | 38 | 1.08% | 1.08% | 1 |
| KO | 38 | 0.85% | 0.84% | 1 |
| LLY | 38 | 1.42% | 1.44% | 3 |
| MA | 38 | 1.18% | 1.19% | 1 |
| META | 36 | 1.86% | 1.86% | 0 |
| MSFT | 38 | 1.42% | 1.41% | 2 |
| NFLX | 38 | 1.55% | 1.55% | 0 |
| NVDA | 38 | 2.25% | 2.19% | 5 |
| ORCL | 38 | 3.33% | 3.29% | 2 |
| PEP | 38 | 1.16% | 1.16% | 0 |
| PG | 38 | 1.13% | 1.13% | 1 |
| TSLA | 38 | 1.96% | 1.96% | 0 |
| UNH | 38 | 1.74% | 1.73% | 1 |
| V | 38 | 1.11% | 1.11% | 0 |
| WMT | 38 | 1.16% | 1.16% | 1 |
| XOM | 38 | 1.31% | 1.33% | 1 |

_Research experiment, not financial advice._
