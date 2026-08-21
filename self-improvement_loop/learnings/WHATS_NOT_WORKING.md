# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 21 of 30 failures (70%) are wrong-direction, not magnitude. The model can't call which way AMZN moves — sizing errors are the minority (9). This is a signal problem, not a calibration-of-magnitude problem.
- **Barely beating a coin flip on direction**: best strategy (news) hits 26.5%, macro just 12%. Aggregate directional skill is negative — a naive persistence baseline would likely do better on many of these days.
- **We lose to baseline constantly**: most failures have ape ≈ baseline_ape (e.g., 06-15, 06-22, 06-29, 07-15, 07-30), meaning the ensemble adds nothing over the trivial forecast on hard days.
- **Big-move days blow up**: 07-31 (13.2% ape) and clustered 3-5% errors (06-22, 06-25, 06-26, 07-23, 08-19) show the model completely fails to anticipate large gaps.

## Unreliable under these conditions
- **Macro-led days are the worst**: winning_strategy=macro went 12% hit rate and produced several of the largest misses (07-15, 07-30, 08-19). Do not trust macro as the deciding vote.
- **High-volatility / large-move sessions**: whenever the actual move is large (>3%), direction is missed and error tracks baseline — the model reverts to mean and gets run over.
- **News-heavy weighting on failure days**: many misses carry news weight 0.30–0.42 (07-13, 07-23, 07-24, 08-11) yet still miss direction — news signal is noisy and over-weighted.
- **Confidence is mildly miscalibrated on the high end**: only 1 flagged overconfident miss, but note 08-19 (conf 0.62) and 08-20 (0.54) were misses, while several low-conf (0.1–0.24) days passed. Confidence has near-zero correlation with being right.

## Fixes to try next
- **Add a volatility regime filter**: on high-vol / gap-risk days, widen intervals and de-weight all trend strategies; stop fighting large moves.
- **Cut macro weight hard** (near zero as a deciding strategy) and cap news weight (~0.20); rebalance toward the least-bad performers.
- **Recalibrate confidence**: current scores don't predict hits — retrain/scale confidence against realized directional accuracy, or collapse to a flat prior until it's informative.
- **Introduce a directional gate**: only take a directional stance when ≥3 strategies agree; otherwise default to persistence baseline, which we're failing to beat.