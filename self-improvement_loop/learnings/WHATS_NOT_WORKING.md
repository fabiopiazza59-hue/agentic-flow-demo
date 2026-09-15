# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Directional errors dominate: 24 of 36 fails (67%) are wrong-direction, not just magnitude. The model gets the *sign* wrong far too often — a coin-flip-or-worse core problem, not a calibration tweak.
- The model rarely beats baseline on fails: nearly every miss also loses to the naive baseline, so it's adding noise, not signal, on bad days.
- Losses cluster on high-move days: big-APE fails (3-5%+, and the 13% outlier on 2026-07-31) are consistently direction misses on volatile sessions the model can't anticipate.
- Overall hit rates are dismal across every strategy (max 29%, news). This is a systemic weakness, not one bad module.

## Unreliable under these conditions
- **macro** is the worst offender: 12% hit rate, highest MAPE (0.0166). It "wins" the ensemble on several of the largest misses (07-13, 07-15, 08-19, 08-25). It should not be steering anything.
- **technical** and **contrarian** are nearly as bad (17% / 18% hit rates). Contrarian repeatedly loses on trending/volatile days (06-26, 08-11, 08-20, 09-03).
- High-volatility / gap days: the June cluster (06-15 through 06-29) and late-July (07-23, 07-30, 07-31) show sustained multi-day failure streaks — the model breaks in choppy regimes.
- **Confidence is miscalibrated on the recent high-conf runs:** conf 0.62–0.72 fails on 08-19, 08-31, 09-03, 09-09, 09-11. When macro/news lead at high confidence, misses persist — overconfidence flagged as 4 but the pattern is broader than counted.

## Fixes to try next
- Cut or gate **macro** entirely; it drags the ensemble and wins on the worst days. Re-weight toward **news** (best hit rate) but cap its downside.
- Attack the directional problem directly: add a regime/volatility filter and abstain (or shrink to baseline) on high-expected-move days rather than committing a sign.
- Recalibrate confidence — high confidence (>0.6) currently does NOT predict success; suppress trades/scores where macro or contrarian is the winning strategy at high conf.
- On days where the model can't beat baseline in backtest conditions, default to baseline instead of the ensemble.