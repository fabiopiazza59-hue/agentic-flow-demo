# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 26 of 41 fails (63%) are wrong-direction, not just magnitude. The model can't call turns, it drifts with the last move.
- Fail rate is 55% (41/74) — worse than a coin flip on direction. Mean fail APE 2.41% is a real miss, not noise.
- Errors cluster on **high-volatility days**: the largest APEs (7-31, 13.2%; 6-22, 5.1%; 6-17, 3.7%; 8-19/8-20, ~2.5%) are almost all fails, and the model rarely beats baseline when moves are big.
- **Macro is the worst strategy** (12% hit, 0.017 MAPE) yet keeps getting picked as winner on fail days (7-13, 7-15, 8-19, 8-25). It's actively harming when it wins.
- **Technical** is also weak (15% hit) but carries a high weight_hint (0.21) — over-weighted relative to its performance.

## Unreliable under these conditions
- **Big-move / gap days** (APE >2.5%): near-universal directional miss regardless of strategy — likely earnings/macro-event days the model has no signal for.
- **Macro-led and contrarian-led predictions on volatile days**: contrarian fades moves that keep running (8-11, 8-20, 9-22, 9-03).
- **Confidence is mildly miscalibrated on the high end**: 8-31 (conf 0.72), 9-09 (0.72), 8-19 (0.62), 9-03 (0.62) all failed. High confidence is not tracking accuracy — several 0.6+ calls miss.
- Low-confidence calls (0.1–0.3) are noisy but sometimes pass — confidence carries little information overall.

## Fixes to try next
- **Cut macro weight toward zero** or gate it to confirmed macro-event days only; stop letting it "win" arbitration.
- **Down-weight technical** to match its 15% hit rate; let news (31% hit, best MAPE) lead.
- **Add a volatility/event filter**: on expected big-move days, widen intervals and drop confidence rather than committing to a direction.
- **Recalibrate confidence** against realized hit rate — current 0.6+ band is overconfident; shrink toward base rate.
- **Fix the directional bias** first — a persistence/mean-reversion regime detector would help since 63% of losses are sign errors.