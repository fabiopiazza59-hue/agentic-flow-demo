# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional errors dominate**: 19 of 28 fails are direction misses (68%). The system can't call the sign, not just the size. This is the core problem, not magnitude.
- **We barely beat a naive baseline.** On most fails `ape ≈ baseline_ape` (e.g. 06-15, 06-22, 06-25, 07-15), meaning the ensemble adds nothing over the previous-close guess and sometimes underperforms it.
- **Blowups on gap days**: 07-31 (ape 13.2%), 06-22 (5.1%), 07-23 (4.2%), 07-30 (3.9%) — clustered large-move days where the model completely fails to size the move.
- **Overconfidence flag reads 0, but calibration is still off**: passes and fails share the same confidence band (~0.4–0.5). Confidence 0.5–0.58 fails (06-12, 06-30, 07-13, 07-20, 08-07) while confidence 0.1 sometimes passes (07-27, 08-03). Confidence carries almost no signal.

## Unreliable under these conditions
- **High-volatility / large-move days**: whenever the true move exceeds ~2–3%, direction and magnitude both collapse. Small quiet days are where nearly all passes live.
- **macro-led predictions are the worst**: 11.4% hit rate, highest MAPE (0.019). Every macro-winning day here (06-12 region, 07-13, 07-15, 07-30) failed or misfired.
- **technical-led is nearly as bad**: 15.9% hit rate despite the 2nd-highest weight hint (0.21) — overweighted relative to performance.
- **news-heavy weighting into event days backfires**: high news weight (0.40–0.42) days (07-13, 07-23, 08-03, 07-24) repeatedly miss direction/magnitude.

## Fixes to try next
- **Cut macro and technical weight**, raise news/momentum (best hit rates: 0.27 / 0.25). Current weight_hints are inverted vs. realized skill.
- **Add a volatility regime gate**: on high-expected-move days, widen predicted magnitude and de-trust the point estimate — the model systematically under-sizes big moves.
- **Recalibrate confidence entirely** — it's uncorrelated with outcomes; rebuild it from realized hit rate per regime/strategy, not model self-report.
- **Attack direction first**: build/track a standalone sign classifier; 68% of fails are sign errors, so magnitude tuning is secondary.