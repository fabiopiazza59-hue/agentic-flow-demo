# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem, not magnitude.** 23 of 35 fails are directional misses (66%). The model calls the wrong way more than it under/over-shoots. On many days ape barely beats baseline even on "passes" — you're tracking, not predicting.
- **The whole ensemble is a coin-flip on direction.** Best strategy (news) hits only 27%; macro is 13%. All five sit below 28% directional hit rate — that's worse than random. The blend has no directional edge.
- **On large-move days it collapses.** 6/22, 6/25, 6/26, 7/15, 7/30, 7/31 (13% ape!), 8/19 — big-APE days are almost all misses and almost all wrong-direction. The model reverts to a small drift and gets run over on volatility spikes.
- **Barely beats baseline anywhere.** Mean fail APE 2.52%, and passes often lose to baseline (beats_baseline=false on many "passes"). Value-add over naive carry-forward is thin.

## Unreliable under these conditions
- **macro as winning_strategy = red flag.** When macro wins (6/12? no — 7/13, 7/15, 7/30, 8/19, 8/25), it's a fail most of the time. macro should not be driving.
- **High-volatility / gap days** (ape > 2.5%): systematically wrong direction.
- **Overconfidence is emerging late.** Recent high-confidence calls miss: 8/19 (conf .62, fail), 8/20 (.54, fail), 9/03 (.62, fail), 8/31 (.72, fail), 9/09 (.72, fail). The 4 flagged overconfident misses are clustered in the last 3 weeks — calibration is drifting worse, not better.
- **News-heavy weightings** don't reliably help direction despite news being the "best" strategy; weight ≠ hit.

## Fixes to try next
- Stop optimizing APE; optimize **directional accuracy** directly. Add a sign-classification head with its own loss.
- **Demote/gate macro** — its winner days lose. Cap macro weight or use only as regime filter.
- Add a **volatility regime detector**; on high-expected-move days widen intervals and lower confidence rather than emitting a tight small-drift point estimate.
- **Recalibrate confidence** — recent .6–.72 calls are missing. Refit confidence-to-hit mapping on the last 20 days; penalize confidence when regime is volatile.
- Investigate the 7/31 13% ape outlier (split/earnings/data error?) before it poisons weight_hints.