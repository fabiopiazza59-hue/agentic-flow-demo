# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 24 of 37 failures (65%) are wrong-direction, not magnitude. The model is guessing the sign wrong, not just the size — this is the core problem.
- **Failures cluster on big-move days**: fail APE averages 2.48% vs sub-0.5% on many passes. When AMZN moves hard (0.03–0.13 APE days: 06-22, 07-15, 07-23, 07-31, 08-19), the model is consistently late and on the wrong side.
- **Barely beating baseline**: many "passes" only tie a naive baseline (06-16, 07-14, 08-27/28 baseline_ape=0). The ensemble adds little on quiet days and hurts on volatile ones.
- **News-weighted blend leads into the worst misses**: heavy news weight (≥0.30) appears in a large share of directional failures (07-13, 07-24, 08-11, 09-11).

## Unreliable under these conditions
- **High-volatility / gap days**: every large-APE failure is a regime shift the ensemble doesn't anticipate. All five strategies collapse together (correlated errors).
- **Macro strategy is the weakest link**: hit_rate 12%, highest MAPE (0.0168). When macro "wins" it usually loses (07-13, 07-15, 08-19, 07-30). Technical (16.7%) and contrarian (18.2%) also below coin-flip.
- **News is best but still only 29% hit_rate** — nothing here is reliably directional; the whole panel is near-random on sign.
- **Confidence is mildly miscalibrated at the top end**: only 4 overconfident misses, but note 08-19 (conf 0.62, fail), 09-03 (0.62, fail), 09-09/09-11 (0.72/0.54, fail). High confidence ≥0.6 does NOT guarantee a pass; several 0.62+ predictions miss.

## Fixes to try next
- **Add a volatility/regime filter**: detect gap/high-range setups and widen intervals or abstain rather than forcing a directional call.
- **Cut or gate macro**: drop its weight toward zero except in confirmed macro-event windows; it's actively diluting the blend.
- **Recalibrate confidence**: high-conf misses show the score isn't tracking accuracy — refit confidence against realized hit-rate, especially the 0.6+ bucket.
- **Attack sign, not magnitude**: since 65% of misses are directional, add a dedicated trend/direction classifier and only defer to magnitude blend once direction agrees.
- **Stop rewarding baseline ties**: tighten the pass threshold so beating a naive carry-forward is required, not optional.