# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional error dominates**: 22 of 33 fails (67%) are direction misses, not magnitude. The model isn't just sizing wrong — it's picking the wrong sign of the next move. Magnitude-only misses are just 11.
- **Baseline is barely beaten**: many fails have ape ≈ baseline_ape (e.g. 06-22, 06-29, 07-15, 08-20). On hard days the ensemble tracks the naive baseline instead of adding edge.
- **Big-move days blow up**: the worst apes (0.13 on 07-31, 0.05 on 06-22, 0.04 on 06-15/07-23, 0.037 on 06-17) cluster on high-volatility days. Mean fail ape 2.58% means the misses are large, not marginal.
- **Macro is the weakest anchor**: whenever macro is the winning strategy it tends to fail (06-12, 07-13, 07-15, 07-30, 08-19 all fails). Macro hit_rate 0.12, worst mape (0.0173).

## Unreliable under these conditions
- **High-volatility / gap days**: model systematically under-reacts, missing direction on large single-day moves.
- **macro-led predictions**: lowest hit rate (12%) and highest error — actively harmful when it wins the vote.
- **technical-led** also weak in direction (15.8% hit) despite low mape — it's precise but pointed the wrong way.
- **Confidence is roughly noise, not calibrated**: only 2 overconfident misses flagged, but high-conf fails exist (08-19 conf 0.62, 08-31 conf 0.72, 07-13 conf 0.55). Low-conf passes (07-27 conf 0.10) also occur. Confidence carries little signal about hit probability.
- **news** is the least-bad strategy (28% hit, lowest mape) but still misses direction >70% of the time — the whole ensemble is a coin-flip at best on direction.

## Fixes to try next
- **Down-weight or gate macro** near zero; it lowers accuracy every time it dominates. Shift weight toward news/momentum.
- **Add a volatility regime detector**: on high-vol/gap days widen predicted magnitude and reduce mean-reversion assumptions; these days drive the tail losses.
- **Rebuild confidence calibration** — current scores don't separate hits from misses. Recalibrate against realized hit-rate; suppress trades when true edge is low.
- **Attack the directional-sign problem directly**: train/evaluate a separate sign classifier rather than relying on magnitude ensemble to infer direction.
- **Blend explicitly against baseline**: since you rarely beat it on hard days, shrink toward baseline when strategy dispersion is high.