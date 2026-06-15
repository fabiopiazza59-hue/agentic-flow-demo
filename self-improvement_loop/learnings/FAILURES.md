# Failure Log — AMZN Close Predictor

Every missed prediction (>1% error), newest at the bottom. Reread before each shot.

## 2026-06-12 — FAIL (APE 2.01%)
- Predicted 243.35 vs actual 238.55 (prior 241.51); dir hit: False; beat baseline: False; closest analyst: momentum.
- **What went wrong:** wrong direction; final blend 243.35 missed by 2.01%.
- **Likely culprit:** analyst `contrarian` was furthest from actual and pulled the blend.
- **Try next:** reduce weight on `contrarian` under today's conditions and lean on `momentum`.
