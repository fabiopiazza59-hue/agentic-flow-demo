# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate.** 24 of 37 fails are wrong-direction, not just magnitude. The model can't call the sign of the move, which is the fatal problem — it's not a calibration issue, it's a signal issue.
- **Failure rate is 55% (37/67), worse than a coin flip.** Mean fail APE 2.48% is large for daily AMZN moves.
- **On big-move days the model gets steamrolled.** APEs of 3–5%+ (2026-06-22, 06-25, 07-15, 07-30, 07-31 at 13%!) cluster where baseline is also large — i.e., high-volatility/gap days where every strategy is blind and just tracks a stale prior.
- **"macro" wins are toxic.** When macro is the winning strategy it repeatedly fails hard (06-12, 07-13, 07-15, 07-30, 08-19). Lowest hit rate (0.134) of any strategy.
- **News is over-weighted relative to its edge.** Highest weight_hint (0.223) and best hit rate (0.284) — but 0.284 is still terrible, and news "wins" often on days it still fails to beat threshold (07-23 4.2% APE, 08-31, 09-09, 09-11).

## Unreliable under these conditions
- **High-volatility / gap days:** all strategies collapse; predictions revert to a lagged level and miss both sign and size.
- **When macro or contrarian is the selected winner:** combined hit rates ~0.13–0.18; these picks are essentially noise.
- **Mid-confidence band (0.4–0.55):** the bulk of misses live here (06-25 .45, 07-13 .55, 08-19 .62, 08-20 .54). Confidence is weakly informative — several 0.5+ predictions miss direction, and 4 flagged overconfident misses confirm the top band isn't earned.
- **Empty-weight days** (07-22, 07-24, 07-30) skew low-confidence and still fail — fallback path is broken.

## Fixes to try next
- **Add a volatility gate:** on high-expected-range days, widen/abstain rather than emit a confident point estimate. Big-move days are where damage concentrates.
- **Cut macro weight toward zero** and cap contrarian; reallocate to news/momentum, but don't trust the ensemble to fix a 55% sign-miss.
- **Attack the directional problem directly:** train/score a separate sign classifier; magnitude tuning is wasted while sign is wrong 24/37 times.
- **Recalibrate confidence** against realized hits — current 0.5–0.7 outputs are not distinguishable from 0.4. Demote until high-confidence bucket actually outperforms.
- **Fix the empty-weights fallback** so it doesn't silently emit low-quality low-confidence guesses.