# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run from the full scored history._

## What keeps going wrong
- 67 of 118 scored predictions FAILED (>1% error); mean error on misses 2.5%.
- Directional misses: 33; magnitude-only misses: 34.
- Overconfident misses (conf ≥ 0.6): 7.

## Unreliable under these conditions
- Highest-MAPE strategy so far: `macro`.
- (Offline summary — richer pattern analysis runs when an API key is set.)

## Fixes to try next
- Tighten magnitude/sizing; down-weight `macro` until its MAPE improves.
- Lower confidence on uncertain setups (calibration).
