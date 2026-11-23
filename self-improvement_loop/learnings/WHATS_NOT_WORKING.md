# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run from the full scored history._

## What keeps going wrong
- 66 of 115 scored predictions FAILED (>1% error); mean error on misses 2.49%.
- Directional misses: 33; magnitude-only misses: 33.
- Overconfident misses (conf ≥ 0.6): 7.

## Unreliable under these conditions
- Highest-MAPE strategy so far: `macro`.
- (Offline summary — richer pattern analysis runs when an API key is set.)

## Fixes to try next
- Tighten direction calls; down-weight `macro` until its MAPE improves.
- Lower confidence on uncertain setups (calibration).
