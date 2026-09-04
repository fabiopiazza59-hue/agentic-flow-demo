# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core failure**: 23 of 34 misses (68%) are directional. The system prices the level roughly right but gets the sign of the move wrong. It's essentially guessing next-day direction.
- **Fails to beat a naive baseline routinely**: many misses have ape ≈ baseline_ape (e.g., 06-17, 06-22, 07-15, 08-20), meaning the model adds nothing over "assume no change."
- **Big-move days are consistently blown**: whenever the true move is large (07-31 ape 13%, 06-22 5%, 07-23 4.2%, 07-30 3.9%), the model badly undershoots. It systematically mean-reverts/dampens and can't anticipate jumps.
- **macro is the worst strategy** (hit rate 11.9%, highest MAPE 0.0172) yet keeps getting chosen as winning strategy on losing days (06-12→? , 07-13, 07-15, 07-30, 08-19, 08-25). When macro drives, it usually loses.

## Unreliable under these conditions
- **High-volatility / gap days**: all worst APEs cluster here; every strategy misses direction together, so ensembling doesn't help.
- **macro- and contrarian-led predictions**: both under 21% hit rate; contrarian misses direction repeatedly (08-11, 08-20, 08-24, 09-03).
- **Rising-confidence regime is miscalibrated**: late-window high-confidence calls (0.62 on 08-19, 0.54 on 08-20, 0.62 on 09-03) are misses. Overconfident_misses=3 undercounts because confidence is compressed low (0.1–0.6); confidence barely correlates with being right.
- **No strategy exceeds 27% directional hit rate** — the whole ensemble is below coin-flip on direction. This is a systemic signal problem, not a single bad strategy.

## Fixes to try next
- **Drop or heavily down-weight macro** (and cap contrarian) until it demonstrates >50% direction; current weight_hints reward losers.
- **Add a volatility gate**: on high-expected-move days, widen intervals and abstain rather than committing a direction the model can't call.
- **Recalibrate confidence** against realized hit rate — current confidence has near-zero discriminative power; force it to track a validated probability.
- **Attack the sign problem directly**: train/evaluate on directional accuracy, not just APE, since magnitude is already near baseline.
- **Investigate the dampening bias**: model systematically underestimates large moves — check for over-smoothing/mean-reversion in features.