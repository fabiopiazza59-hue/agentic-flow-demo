# Failure Log — AMZN Close Predictor

Every missed prediction (>1% error), newest at the bottom. Reread before each shot.

## 2026-06-12 — FAIL (APE 2.01%)
- Predicted 243.35 vs actual 238.55 (prior 241.51); dir hit: False; beat baseline: False; closest analyst: momentum.
- **What went wrong:** wrong direction; final blend 243.35 missed by 2.01%.
- **Likely culprit:** analyst `contrarian` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `contrarian` under today's conditions and lean on `momentum`.

## 2026-06-15 — FAIL (APE 3.40%)
- Predicted 237.65 vs actual 246.02 (prior 238.55); dir hit: False; beat baseline: False; closest analyst: news.
**What went wrong:** We blended toward continuation-down (momentum/macro at 0.60 combined) when every structural signal screamed exhaustion: RSI 26.76, ~7% below SMA20, -10.7% over 20d. Oversold extremes don't continue linearly — they snap back, and a U.S.-Iran peace deal lifting Nasdaq futures ~2% was the exact catalyst news flagged. **Root cause:** the meta-weights rewarded trend-following analysts in a regime where trend-following is structurally wrong (deep oversold + exogenous positive catalyst), and news — despite the best track record (MAPE 1.06%, weight_hint 0.31) — was under-weighted at 0.12. We trusted recency-of-trend over signal-of-reversal. **One change:** add a regime gate — when RSI < 30 and a fresh market-moving catalyst is detected, cut momentum+macro weight in half and floor news+contrarian at the blend majority. Had we done so here, the blend lands near 242-244, halving the error and catching direction.

## 2026-06-17 — FAIL (APE 3.73%)
- Predicted 246.35 vs actual 237.5 (prior 246.0); dir hit: False; beat baseline: False; closest analyst: technical.
- **What went wrong:** wrong direction; final blend 246.35 missed by 3.73%.
- **Likely culprit:** analyst `contrarian` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `contrarian` under today's conditions and lean on `technical`.
