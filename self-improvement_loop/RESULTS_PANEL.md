# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (22 sessions, 659 name-forecasts; mean daily APE gain -0.006%, CS [-0.020%, +0.007%]).

**Pending (2026-10-28):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.006% | [-0.020%, +0.007%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.003% | [-0.013%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.116 | [-0.407, +0.175] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0016 | [-0.005, +0.002] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.018% | [-0.086%, +0.122%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.024% | [-0.077%, +0.126%] | ≈ not distinguishable |

MAPE — LLM 1.40%, free baseline 1.39%, prior close 1.41%; LLM 80% interval coverage 78.62%; names moved per session 1.0. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 17 | -0.002% | [-0.039%, +0.035%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 16 | +0.004% | [-0.031%, +0.039%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 22 | 0.93% | 0.93% | 0 |
| ADBE | 22 | 2.08% | 2.08% | 0 |
| AMD | 22 | 2.43% | 2.43% | 0 |
| AMZN | 22 | 1.42% | 1.42% | 0 |
| AVGO | 22 | 1.94% | 1.91% | 4 |
| BAC | 22 | 0.88% | 0.90% | 1 |
| CAT | 22 | 1.71% | 1.71% | 0 |
| COST | 22 | 0.87% | 0.86% | 1 |
| CRM | 22 | 1.64% | 1.64% | 0 |
| CVX | 22 | 1.10% | 1.10% | 0 |
| DIS | 22 | 0.93% | 0.93% | 1 |
| GOOGL | 22 | 1.20% | 1.20% | 0 |
| HD | 22 | 1.24% | 1.25% | 1 |
| JNJ | 22 | 1.56% | 1.54% | 1 |
| JPM | 22 | 0.74% | 0.72% | 1 |
| KO | 22 | 0.99% | 0.97% | 1 |
| LLY | 22 | 1.42% | 1.40% | 1 |
| MA | 22 | 1.06% | 1.06% | 0 |
| META | 21 | 2.04% | 2.04% | 0 |
| MSFT | 22 | 0.97% | 0.96% | 1 |
| NFLX | 22 | 1.62% | 1.62% | 0 |
| NVDA | 22 | 1.35% | 1.31% | 3 |
| ORCL | 22 | 2.90% | 2.82% | 2 |
| PEP | 22 | 1.14% | 1.14% | 0 |
| PG | 22 | 1.27% | 1.29% | 1 |
| TSLA | 22 | 1.73% | 1.73% | 0 |
| UNH | 22 | 1.47% | 1.45% | 1 |
| V | 22 | 1.00% | 1.00% | 0 |
| WMT | 22 | 1.23% | 1.23% | 1 |
| XOM | 22 | 1.09% | 1.09% | 0 |

_Research experiment, not financial advice._
