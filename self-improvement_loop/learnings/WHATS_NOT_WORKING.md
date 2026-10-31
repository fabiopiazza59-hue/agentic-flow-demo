# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Overall failure rate is high (55/99, 56%) and, critically, **30 of 55 fails are directional** — the model is picking the wrong sign of the daily move more than half the time on misses, not just sizing it wrong.
- The system **barely beats a naive baseline**: many fails have ape ≈ baseline_ape (e.g. 06-22, 07-15, 08-10, 08-20, 10-08). On a large chunk of days the strategies add nothing over the baseline.
- **Large-move days blow up magnitude.** Big APEs (05-19 0.054, 07-31 0.132, 10-19 0.055, 10-30 0.052, 06-22/07-23 ~0.04-0.05) cluster on volatile/gap sessions. mean_fail_ape of 2.42% is dragged by these tails.
- **macro and technical are dead weight**: both hit_rate 0.121 (12/99) with the two worst MAPEs. They "win" the ensemble on some of the worst days (07-13, 07-15, 08-19, 10-30) — i.e. they get upweighted right before big misses.
- **Confidence is creeping up without accuracy.** Recent block (late Sep–Oct) shows confidence 0.55–0.66 on a run of fails (10-06, 10-08, 10-09, 10-19). Overconfident misses (7) are concentrated in high-vol fails like 08-19 (0.62), 08-20 (0.54), 10-19 (0.56).

## Unreliable under these conditions
- **High-volatility / gap days (ape > 3%)**: directional AND magnitude both fail; these are ~1/3 of total error mass.
- **When macro or technical is the winning strategy**: disproportionately associated with fails and worst APEs.
- **Contrarian on trending days**: 25 wins but also drives several directional misses (10-08, 10-15, 10-19, 09-22) — it fades real moves.
- **Rising-confidence regime (0.55+)**: calibration breaks down; confidence no longer tracks pass probability.

## Fixes to try next
- Cut or floor **macro and technical** weights (both 12% hit rate); shift mass toward **news** (33% hit, best MAPE) with contrarian as secondary.
- Add a **volatility/gap detector**: on high-vol days widen intervals, drop confidence, and suppress contrarian (stop fading breakouts).
- **Recalibrate confidence** against realized pass rate — current high-confidence band is anti-predictive; cap confidence until Brier score improves.
- Build a **directional guardrail**: since 30/55 fails are sign errors, add a trend/sign filter before committing to a contrarian call.
- Track beats_baseline as a gating metric — kill any ensemble day that can't demonstrably beat the naive baseline in backtest.