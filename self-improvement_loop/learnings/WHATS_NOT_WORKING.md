# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (30 of 51 fails). The model gets the *sign* of the move wrong more than magnitude — this is the core problem, not calibration of size.
- **We barely beat the coin flip.** Best strategy (news) hits direction only 31.5%; macro (12%) and technical (13%) are worse than random. The ensemble is built on mostly-wrong directional signals.
- **Tail events wreck us.** 2026-07-31 (13.2% APE) and several ~3-5% APE days (06-22, 07-23, 10-19) show no ability to anticipate large moves; we track the baseline into the hole and barely edge it.
- **Macro and contrarian lead on bad days.** A disproportionate share of big-APE fails list macro or contrarian as winning_strategy — they win the internal vote precisely when they're wrong.

## Unreliable under these conditions
- **High-volatility / gap days** (APE > 3%): clustered in mid-to-late June and late July. Model has no regime detector and keeps applying mean-reversion logic into trending breaks.
- **Macro-driven sessions:** macro winning_strategy correlates with the worst outcomes (07-13, 07-15, 08-19, 10-09). Macro signal is noise.
- **Rising confidence, flat accuracy (recent regime, Oct):** confidence climbed to 0.58-0.66 while fails continued (10-06, 10-07, 10-08, 10-19). The 7 overconfident misses cluster where conf ≥ 0.6 but direction wrong — **calibration is drifting upward without earned accuracy.**
- **Contrarian in Oct:** repeatedly the winning strategy on fails (10-08, 10-12, 10-14, 10-15, 10-19) — it's firing counter-trend in a trending tape.

## Fixes to try next
- **Drop or heavily down-weight macro and technical** (hit rates ≤13%); rebalance toward news, cap any single strategy's vote.
- **Add a volatility/regime gate:** when realized vol or overnight gap exceeds threshold, widen intervals and suppress contrarian.
- **Recalibrate confidence** — recent conf ≥0.6 is not justified; regress confidence onto actual hit rate and shrink toward 0.5.
- **Directional-first scoring:** retune ensemble to optimize sign accuracy, not just APE, since magnitude is the smaller problem.
- **Suppress contrarian during confirmed trends** (e.g., gate on multi-day momentum sign agreement).