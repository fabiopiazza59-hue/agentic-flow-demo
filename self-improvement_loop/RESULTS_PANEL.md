# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (7 sessions, 209 name-forecasts; mean daily APE gain 0.002%, CS n=7 sessions (too few)).

**Pending (2026-10-06):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.002% | n=7 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.001% | n=7 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.001 | n=7 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.030% | n=7 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.032% | n=7 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.15%, free baseline 1.15%, prior close 1.12%; LLM 80% interval coverage 80.85%; names moved per session 0.6. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 2 | -0.015% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 1 | -0.033% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 7 | 1.00% | 1.00% | 0 |
| ADBE | 7 | 1.21% | 1.21% | 0 |
| AMD | 7 | 1.49% | 1.49% | 0 |
| AMZN | 7 | 0.85% | 0.85% | 0 |
| AVGO | 7 | 2.20% | 2.20% | 0 |
| BAC | 7 | 0.94% | 0.94% | 0 |
| CAT | 7 | 1.23% | 1.23% | 0 |
| COST | 7 | 0.66% | 0.64% | 1 |
| CRM | 7 | 1.97% | 1.97% | 0 |
| CVX | 7 | 0.62% | 0.62% | 0 |
| DIS | 7 | 0.98% | 1.00% | 1 |
| GOOGL | 7 | 1.07% | 1.07% | 0 |
| HD | 7 | 0.97% | 1.00% | 1 |
| JNJ | 7 | 1.26% | 1.26% | 0 |
| JPM | 7 | 0.79% | 0.79% | 0 |
| KO | 7 | 0.59% | 0.59% | 0 |
| LLY | 7 | 0.80% | 0.80% | 0 |
| MA | 7 | 1.14% | 1.14% | 0 |
| META | 6 | 2.02% | 2.02% | 0 |
| MSFT | 7 | 0.78% | 0.78% | 0 |
| NFLX | 7 | 1.77% | 1.78% | 0 |
| NVDA | 7 | 0.95% | 0.95% | 0 |
| ORCL | 7 | 1.90% | 1.90% | 0 |
| PEP | 7 | 0.48% | 0.48% | 0 |
| PG | 7 | 1.30% | 1.30% | 0 |
| TSLA | 7 | 1.78% | 1.78% | 0 |
| UNH | 7 | 1.10% | 1.10% | 0 |
| V | 7 | 0.81% | 0.81% | 0 |
| WMT | 7 | 1.28% | 1.31% | 1 |
| XOM | 7 | 0.63% | 0.63% | 0 |

_Research experiment, not financial advice._
