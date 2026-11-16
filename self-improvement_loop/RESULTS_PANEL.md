# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (36 sessions, 1078 name-forecasts; mean daily APE gain -0.002%, CS [-0.014%, +0.010%]).

**Pending (2026-11-17):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.014%, +0.010%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.010%, +0.008%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0012 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.086% | [-0.051%, +0.223%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.088% | [-0.044%, +0.221%] | ≈ not distinguishable |

MAPE — LLM 1.58%, free baseline 1.58%, prior close 1.67%; LLM 80% interval coverage 78.94%; names moved per session 1.1. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 31 | -0.012% | [-0.044%, +0.020%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 30 | -0.003% | [-0.034%, +0.028%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 36 | 1.16% | 1.15% | 1 |
| ADBE | 36 | 1.77% | 1.77% | 0 |
| AMD | 36 | 3.71% | 3.74% | 3 |
| AMZN | 36 | 1.77% | 1.77% | 0 |
| AVGO | 36 | 2.47% | 2.47% | 5 |
| BAC | 36 | 1.05% | 1.06% | 1 |
| CAT | 36 | 2.40% | 2.38% | 1 |
| COST | 36 | 0.95% | 0.94% | 1 |
| CRM | 36 | 1.45% | 1.45% | 0 |
| CVX | 36 | 1.21% | 1.22% | 1 |
| DIS | 36 | 1.03% | 1.02% | 2 |
| GOOGL | 36 | 1.33% | 1.35% | 2 |
| HD | 36 | 1.38% | 1.39% | 1 |
| JNJ | 36 | 1.49% | 1.48% | 1 |
| JPM | 36 | 1.02% | 1.01% | 1 |
| KO | 36 | 0.88% | 0.87% | 1 |
| LLY | 36 | 1.44% | 1.46% | 3 |
| MA | 36 | 1.19% | 1.19% | 1 |
| META | 34 | 1.94% | 1.94% | 0 |
| MSFT | 36 | 1.44% | 1.43% | 2 |
| NFLX | 36 | 1.59% | 1.59% | 0 |
| NVDA | 36 | 2.15% | 2.09% | 5 |
| ORCL | 36 | 3.31% | 3.27% | 2 |
| PEP | 36 | 1.16% | 1.16% | 0 |
| PG | 36 | 1.16% | 1.17% | 1 |
| TSLA | 36 | 2.03% | 2.03% | 0 |
| UNH | 36 | 1.40% | 1.38% | 1 |
| V | 36 | 1.12% | 1.12% | 0 |
| WMT | 36 | 1.19% | 1.20% | 1 |
| XOM | 36 | 1.26% | 1.28% | 1 |

_Research experiment, not financial advice._
