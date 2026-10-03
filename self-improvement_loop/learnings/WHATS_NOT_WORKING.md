# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Directional misses dominate (27 of 42 fails, 64%). The model is getting the *direction* wrong more than the magnitude — this is a sign problem, not a precision problem.
- Fails cluster on high-volatility days: mean fail APE 2.39% and the worst days (2026-07-31 at 13.2%, 06-22 at 5.1%, 07-23 at 4.2%, 06-17/06-25/07-15/07-30 at 3-4%) are all large-move sessions the model smooths over.
- On big-move days the model barely beats or loses to the naive baseline (06-12, 06-15, 06-17, 06-22, 06-25, 07-13 all fail AND lose to baseline). Edge collapses exactly when it matters.
- `macro` is a persistent loser: 11.4% hit rate, highest MAPE (0.0164), yet it still "wins" the ensemble on volatile days (06-08, 07-13, 07-15, 07-30, 08-19) and drags accuracy down.
- `technical` is nearly as bad (13.9% hit rate) despite carrying the 2nd-highest weight hint (0.208). Weighting is inversely correlated with reliability.

## Unreliable under these conditions
- Large daily moves / volatility spikes: both direction and magnitude break down together.
- Any day the ensemble lands on `macro` or `technical` as the winner — low hit rates, high error.
- Recent confidence inflation: late-period confidence runs 0.52-0.72 while pass rate is mixed; overconfident misses (08-19 conf 0.62, 08-31 conf 0.72, 09-09 conf 0.72, 09-11 conf 0.54, 09-03 conf 0.62) show confidence rising without accuracy rising. Calibration is drifting high.
- `weights: {}` / `null` days (06-30, 07-22, 07-28, 08-05, 08-13, 09-17, 09-22, and all post-09-28) correlate with fails — ensemble config appears broken/unlogged in those windows.

## Fixes to try next
- Cut `macro` and `technical` weights sharply (or gate them off on high-vol days); reallocate to `news` (best hit rate 0.329, lowest MAPE).
- Add a volatility regime filter: widen intervals / lower confidence when expected range is large, since that's where directional sign flips.
- Recalibrate confidence against realized outcomes — current high-confidence late-period predictions are not earning it.
- Fix the null/empty-weights logging path; those days are untraceable and skew toward failure.
- Build a direction-only sub-model or sign-check layer, since 64% of fails are directional, not magnitude.