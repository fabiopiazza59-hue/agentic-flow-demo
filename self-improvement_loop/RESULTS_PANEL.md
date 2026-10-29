# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
❌ **No edge yet** — the LLM's gain over the free baseline is not distinguishable from noise (23 sessions, 689 name-forecasts; mean daily APE gain -0.006%, CS [-0.019%, +0.007%]).

**Pending (2026-10-29):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 5 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | -0.006% | [-0.019%, +0.007%] | ≈ not distinguishable |
| CRPS gain: LLM vs free baseline | -0.003% | [-0.012%, +0.006%] | ≈ not distinguishable |
| Rank IC (adjustment vs realized move off the anchor) | -0.116 | [-0.407, +0.175] | ≈ not distinguishable |
| Direction (Brier of P(up)) vs free baseline | -0.0015 | [-0.005, +0.002] | ≈ not distinguishable |
| APE gain: LLM vs prior close | 0.017% | [-0.082%, +0.115%] | ≈ not distinguishable |
| APE gain: anchor vs prior close (the free information itself) | 0.023% | [-0.073%, +0.119%] | ≈ not distinguishable |

MAPE — LLM 1.39%, free baseline 1.38%, prior close 1.40%; LLM 80% interval coverage 79.40%; names moved per session 0.9. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 18 | -0.002% | [-0.036%, +0.033%] (α=0.0250) | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 17 | +0.003% | [-0.030%, +0.036%] (α=0.0083) | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 23 | 0.93% | 0.93% | 0 |
| ADBE | 23 | 2.03% | 2.03% | 0 |
| AMD | 23 | 2.43% | 2.43% | 0 |
| AMZN | 23 | 1.37% | 1.37% | 0 |
| AVGO | 23 | 1.96% | 1.94% | 4 |
| BAC | 23 | 0.85% | 0.87% | 1 |
| CAT | 23 | 1.64% | 1.64% | 0 |
| COST | 23 | 0.87% | 0.87% | 1 |
| CRM | 23 | 1.61% | 1.61% | 0 |
| CVX | 23 | 1.08% | 1.08% | 0 |
| DIS | 23 | 0.93% | 0.94% | 1 |
| GOOGL | 23 | 1.16% | 1.16% | 0 |
| HD | 23 | 1.22% | 1.23% | 1 |
| JNJ | 23 | 1.53% | 1.51% | 1 |
| JPM | 23 | 0.75% | 0.74% | 1 |
| KO | 23 | 0.97% | 0.96% | 1 |
| LLY | 23 | 1.38% | 1.35% | 1 |
| MA | 23 | 1.06% | 1.06% | 0 |
| META | 22 | 2.00% | 2.00% | 0 |
| MSFT | 23 | 0.94% | 0.93% | 1 |
| NFLX | 23 | 1.59% | 1.59% | 0 |
| NVDA | 23 | 1.35% | 1.31% | 3 |
| ORCL | 23 | 3.16% | 3.09% | 2 |
| PEP | 23 | 1.12% | 1.12% | 0 |
| PG | 23 | 1.24% | 1.26% | 1 |
| TSLA | 23 | 1.70% | 1.70% | 0 |
| UNH | 23 | 1.44% | 1.42% | 1 |
| V | 23 | 0.99% | 0.99% | 0 |
| WMT | 23 | 1.19% | 1.20% | 1 |
| XOM | 23 | 1.09% | 1.09% | 0 |

_Research experiment, not financial advice._
