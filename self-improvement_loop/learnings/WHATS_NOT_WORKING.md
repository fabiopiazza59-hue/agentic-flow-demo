# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem.** 23 of 35 failures (66%) are directional misses, not magnitude. The model gets the size roughly right but bets the wrong way — a sign-prediction failure, not a scaling one.
- **Weak overall hit rate.** Best strategy (news) hits direction only 27%; macro is a coin-flip-losing 13%. No strategy clears 30%. The ensemble is effectively worse than random on direction.
- **Barely beats baseline.** Many fails have ape ≈ baseline_ape (e.g. 06-15, 06-22, 07-15, 08-20). On losing days the model isn't adding value over naive persistence — it's just tracking it and occasionally overshooting.
- **Big-move blindness.** On high-volatility days (07-31 ape 13%, 06-22 5%, 07-23 4%, 06-17 3.7%) the model badly under/over-shoots. Tail days dominate mean_fail_ape (2.52%).

## Unreliable under these conditions
- **macro-led and contrarian-led days.** When winning_strategy is macro, results are consistently poor (12.7% hit rate); contrarian is nearly as bad. These two are dragging the ensemble.
- **Large-gap / gap-open days.** The worst APEs cluster on days with big baseline moves — the model fails to size or direction regime shifts.
- **Rising confidence late in sample is NOT earned.** Confidence climbed to 0.62–0.72 in Sept, yet 08-19 (0.62), 08-31 (0.72), 09-03 (0.62), 09-09 (0.72) all failed. Overconfident_misses is only tagged 4, but calibration is drifting up while accuracy isn't — confidence is becoming decorative.
- **Low-confidence days aren't safer either** (07-30 conf 0.1 failed at 3.9%), so confidence carries almost no discriminative signal.

## Fixes to try next
- **Cut macro and contrarian weight** (raise news/technical) or gate them to specific regimes; they're the lowest-hit, highest-mape strategies.
- **Attack direction directly:** add a separate sign classifier / regime filter; don't let magnitude models set direction.
- **Recalibrate confidence** against realized hit rate — current high-confidence Sept predictions are miscalibrated; apply isotonic/Platt scaling and shrink toward 0.5.
- **Add a volatility-aware widening** on flagged high-move days; the tail misses (07-31) alone distort the mean.
- **Set an abstain/baseline-fallback** rule when ensemble disagreement is high, since on those days it merely matches persistence.