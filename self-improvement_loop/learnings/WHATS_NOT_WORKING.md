# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Failure rate is 57% (61/107)** — the system loses more than it wins. Misses are split near-evenly: 32 directional vs 29 magnitude, so there's no single lever.
- **Big-move days are the killer.** Every blow-up (APE 3–13%) clusters on high-volatility sessions: 2026-07-31 (13.2%), 06-22 (5.1%), 10-19 (5.5%), 11-04 (6.7%), 10-30 (5.2%). The model systematically underestimates magnitude on gap/event days and often gets direction wrong too.
- **macro and technical are dead weight.** Hit rates 12% and 11% respectively, highest MAPE (0.0173, 0.0156). When either "wins" the ensemble, you usually lose (e.g., 07-13, 07-15, 08-19, 10-30).
- **Recent regime (late Oct–Nov) is deteriorating:** ~10 of last 15 are fails, many beating baseline directionally but still missing the APE threshold — magnitude scaling is too timid in trending/volatile tape.

## Unreliable under these conditions
- **High-volatility / earnings-adjacent days** (any session with true move >2.5%): both direction and magnitude collapse.
- **macro-led and technical-led predictions**: avoid as the deciding strategy; they underperform the 0.5 coin-flip baseline on direction.
- **Mid-range confidence 0.5–0.72**: 7 overconfident misses sit here (08-19 @0.62, 08-31 @0.72, 09-09 @0.72, 10-06 @0.66). Confidence rises but accuracy doesn't — **calibration is broken above ~0.55**.
- **Contrarian in sustained trends**: 27% hit rate but produces the largest tail errors (10-19, 11-04) when it fades a real move.

## Fixes to try next
- **Downweight macro+technical hard** (cap each ≤0.08) or gate them out; shift weight to news (0.32 hit, lowest MAPE) and contrarian for mean-reversion only.
- **Add a volatility regime switch**: on expected-high-move days, widen magnitude and suppress contrarian (it fades real breakouts).
- **Recalibrate confidence**: cap output at ~0.55 until high-confidence buckets demonstrate >60% accuracy; current 0.6–0.72 band is anti-predictive.
- **Separate directional vs magnitude scoring** in tuning — fixing one isn't fixing the other right now.