# 📊 Panel — does an LLM add value over the pre-open price?

_30 US large caps a session. Every name's free baseline is its latest pre-open trade (the anchor); the LLM may move it by at most ±0.5σ. Verdicts average each metric over the names of a session and read an anytime-valid 95% confidence sequence on that daily series, so they stay valid although this page is regenerated daily. Pre-open forecasts only. Design: [spec/panel.md](spec/panel.md)._

## Verdict
⏳ **Not enough clean sessions yet** to judge the LLM against the free baseline (8 sessions, 239 name-forecasts; mean daily APE gain 0.002%, CS n=8 sessions (too few)).

**Pending (2026-10-08):** 30 names, 30 anchored on a live pre-open trade, the LLM moved 0 off their anchor.

| Daily metric (mean over names) | Mean | 95% CS | Verdict |
|---|---|---|---|
| APE gain: LLM vs free baseline | 0.002% | n=8 sessions (too few) | ⏳ too few sessions |
| CRPS gain: LLM vs free baseline | 0.001% | n=8 sessions (too few) | ⏳ too few sessions |
| Rank IC (adjustment vs realized move off the anchor) | 0.075 | n=1 sessions (too few) | ⏳ too few sessions |
| Direction (Brier of P(up)) vs free baseline | 0.0009 | n=8 sessions (too few) | ⏳ too few sessions |
| APE gain: LLM vs prior close | -0.027% | n=8 sessions (too few) | ⏳ too few sessions |
| APE gain: anchor vs prior close (the free information itself) | -0.028% | n=8 sessions (too few) | ⏳ too few sessions |

MAPE — LLM 1.13%, free baseline 1.13%, prior close 1.10%; LLM 80% interval coverage 81.61%; names moved per session 0.5. LLM is **on**.

## Prompt evolution — shadow challengers, promoted only on proof

_Challenger prompts run on the same snapshot without shipping. One replaces the champion only when an anytime-valid CS on the session-averaged APE gain excludes zero; the k-th challenger ever created is tested at α/(k(k+1)), so the chance of ever promoting a prompt that is not better stays below α. The fixed output contract and the ±σ clamp live in code; every prompt is in `learnings/prompt_variants.json`._

| Variant | Status | Parent | Sessions | Gain vs champion | CS (α_k) | Note |
|---|---|---|---|---|---|---|
| p0 | champion | — | — | — | — | seed: the hand-written panel prompt |
| p1 | challenger | p0 | 3 | -0.010% | — | The failure cases show positive overnight gaps systematically followed by negative realized moves off the anchor (and vi |
| p2 | challenger | p0 | 2 | -0.019% | — | The champion moved almost nothing (2.2% of names) and realized moves were overwhelmingly opposite the overnight gap, so  |

## Per name (pre-open forecasts)

| Name | Sessions | LLM MAPE | Free-baseline MAPE | Days moved |
|---|---|---|---|---|
| AAPL | 8 | 0.98% | 0.98% | 0 |
| ADBE | 8 | 1.33% | 1.33% | 0 |
| AMD | 8 | 1.42% | 1.42% | 0 |
| AMZN | 8 | 0.88% | 0.88% | 0 |
| AVGO | 8 | 1.96% | 1.96% | 0 |
| BAC | 8 | 0.99% | 0.99% | 0 |
| CAT | 8 | 1.86% | 1.86% | 0 |
| COST | 8 | 0.68% | 0.66% | 1 |
| CRM | 8 | 1.74% | 1.74% | 0 |
| CVX | 8 | 0.69% | 0.69% | 0 |
| DIS | 8 | 0.93% | 0.94% | 1 |
| GOOGL | 8 | 1.02% | 1.02% | 0 |
| HD | 8 | 0.90% | 0.93% | 1 |
| JNJ | 8 | 1.27% | 1.27% | 0 |
| JPM | 8 | 0.78% | 0.78% | 0 |
| KO | 8 | 0.60% | 0.60% | 0 |
| LLY | 8 | 1.01% | 1.01% | 0 |
| MA | 8 | 1.06% | 1.06% | 0 |
| META | 7 | 1.80% | 1.80% | 0 |
| MSFT | 8 | 0.68% | 0.68% | 0 |
| NFLX | 8 | 1.64% | 1.65% | 0 |
| NVDA | 8 | 0.96% | 0.96% | 0 |
| ORCL | 8 | 1.80% | 1.80% | 0 |
| PEP | 8 | 0.64% | 0.64% | 0 |
| PG | 8 | 1.18% | 1.18% | 0 |
| TSLA | 8 | 1.62% | 1.62% | 0 |
| UNH | 8 | 0.97% | 0.97% | 0 |
| V | 8 | 0.75% | 0.75% | 0 |
| WMT | 8 | 1.23% | 1.26% | 1 |
| XOM | 8 | 0.59% | 0.59% | 0 |

_Research experiment, not financial advice._
