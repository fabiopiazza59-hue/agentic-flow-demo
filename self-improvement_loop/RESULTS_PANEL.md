# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (44 sessions, 1318 name-forecasts; mean daily APE gain -0.002%, CS [-0.011%, +0.008%]).

**Pending (2026-11-30):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.002% | [-0.011%, +0.008%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.001% | [-0.008%, +0.007%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.019 | [-0.239, +0.200] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.001 | [-0.003, +0.001] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.113% | [-0.035%, +0.260%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.114% | [-0.030%, +0.259%] | ≈ not distinguishable |

MAPE — LLM 1.68%, free baseline 1.68%, prior close 1.79%; LLM 80% interval coverage 79.05%; names moved per session 0.9. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 39 | -0.009% | [-0.034%, +0.016%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 38 | -0.003% | [-0.027%, +0.021%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 44 | 1.16% | 1.16% | 1 |
| ADBE | 44 | 1.93% | 1.93% | 0 |
| AMD | 44 | 3.88% | 3.91% | 3 |
| AMZN | 44 | 1.86% | 1.86% | 0 |
| AVGO | 44 | 2.66% | 2.66% | 5 |
| BAC | 44 | 1.09% | 1.10% | 1 |
| CAT | 44 | 2.56% | 2.54% | 1 |
| COST | 44 | 0.91% | 0.91% | 1 |
| CRM | 44 | 1.81% | 1.81% | 0 |
| CVX | 44 | 1.23% | 1.24% | 1 |
| DIS | 44 | 1.17% | 1.16% | 2 |
| GOOGL | 44 | 1.41% | 1.43% | 2 |
| HD | 44 | 1.34% | 1.35% | 1 |
| JNJ | 44 | 1.46% | 1.45% | 1 |
| JPM | 44 | 1.13% | 1.12% | 1 |
| KO | 44 | 0.86% | 0.85% | 1 |
| LLY | 44 | 1.39% | 1.41% | 3 |
| MA | 44 | 1.21% | 1.22% | 1 |
| META | 42 | 2.12% | 2.12% | 0 |
| MSFT | 44 | 1.41% | 1.40% | 2 |
| NFLX | 44 | 1.66% | 1.66% | 0 |
| NVDA | 44 | 2.91% | 2.86% | 5 |
| ORCL | 44 | 3.42% | 3.39% | 2 |
| PEP | 44 | 1.11% | 1.11% | 0 |
| PG | 44 | 1.07% | 1.08% | 1 |
| TSLA | 44 | 2.31% | 2.31% | 0 |
| UNH | 44 | 1.81% | 1.80% | 1 |
| V | 44 | 1.15% | 1.15% | 0 |
| WMT | 44 | 1.16% | 1.16% | 1 |
| XOM | 44 | 1.28% | 1.28% | 1 |

_Research experiment, not financial advice._
