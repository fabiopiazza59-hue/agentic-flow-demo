# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (28 of 47 fails, ~60%). The core problem is getting *sign* wrong, not just magnitude — the model can't call the next day's direction better than a coin flip.
- **Failures cluster on big-move days.** Nearly every high-APE fail (3-5%+, and the 13% blowup on 07-31) is a day where the realized move was large and the model stayed anchored near the prior close. Baseline APE is comparable on these days — we're not adding edge when it matters.
- **Beating baseline is rare on fails.** Many fails also have `beats_baseline=false`, meaning on bad days we're worse than a naive carry-forward, not just unlucky.
- **Macro and technical are dead weight.** Both at 12.8% hit rate and the two highest MAPEs (macro 0.0163, technical 0.0142). Macro-led days are disproportionately large-error fails (07-15, 07-30, 08-19).

## Unreliable under these conditions
- **High-volatility / gap days:** model systematically under-reacts, predicting near-flat while price moves 2-5%.
- **macro as winning_strategy:** 11/86 wins, repeatedly the strategy behind the worst misses. Lowest weight_hint yet still steering bad calls.
- **technical:** same 12.8% hit rate; adds magnitude error without directional value.
- **Rising-confidence late window (Oct, conf 0.59-0.66):** 10-06 through 10-09 are confident fails. **Recent overconfidence is emerging** — the newer regime (conf creeping up) shows passes falling while confidence rises. Miscalibration is directional: confidence is trending up precisely as accuracy degrades.
- **contrarian** carries decent hit rate (23%) but is behind several directional fails on choppy reversals (08-20, 09-22, 10-08) — unreliable when the market is actually trending.

## Fixes to try next
- **Cut macro and technical weights hard** (toward ~0.05-0.10); reallocate to **news** (best: 31% hit, lowest MAPE).
- **Add a volatility regime gate:** when expected/realized range is high, widen prediction away from prior close or defer to the higher-variance strategies instead of anchoring flat.
- **Recalibrate confidence** — recent confidence (0.6+) is not earning its keep; cap confidence on macro/technical-led and high-vol days.
- **Build a dedicated directional model** (sign classifier) since 60% of failures are sign errors, not magnitude.
- **Stop trusting contrarian in trending tapes** — add a trend filter before it can win the ensemble.