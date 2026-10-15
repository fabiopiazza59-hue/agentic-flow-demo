# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate** (28 of 48 fails, 58%). The system gets the *sign* wrong more than the *size* — this is a signal problem, not a calibration-of-magnitude problem.
- **Big-move days are a bloodbath.** On every day with baseline APE > ~2.5% the model also fails and barely beats baseline (e.g. 2026-06-22 5.1%, 2026-07-31 13.2%, 2026-07-30 3.9%, 2026-08-19 2.9%). Mean fail APE is 2.3% vs sub-0.5% on clean passes — errors cluster entirely in high-volatility sessions.
- **Barely-better-than-baseline even when "winning."** Many fails have ape ≈ baseline_ape (within 0.001), meaning the blend adds no edge on hard days — it just tracks the naive baseline into the same wall.
- **Confidence is drifting upward without accuracy.** Recent block (2026-10-06 → 10-14) shows confidence 0.59–0.66 on a run of fails. The 7 overconfident misses are concentrated here and around 2026-08-19/20 (conf 0.62/0.54).

## Unreliable under these conditions
- **High realized volatility / gap days:** both news and macro get selected as "winner" and still miss direction (macro 2026-07-15, 07-30, 08-19; news 06-15, 06-22, 06-25).
- **macro and technical are the weakest engines** — hit rate 12.6% each, macro has the worst MAPE (1.63%). They should not be driving picks yet they win on several of the worst fails.
- **Rising-confidence regime (Oct):** confidence climbing 0.53→0.66 while pass rate falls — calibration breaks exactly when the model feels most sure.
- **contrarian over-weighted on choppy days** (2026-08-20, 09-22, 10-08 all fails with heavy contrarian weight).

## Fixes to try next
- **Cap confidence as a function of recent realized vol**; on high-vol days force wider intervals and lower stated confidence.
- **Demote macro and technical** in the blend (hit rate ~0.13); lean on news (0.31) and contrarian (0.24), but gate contrarian off during trending/high-vol sessions.
- **Add a volatility regime flag**: if expected range > ~2%, switch to a defensive ensemble or abstain rather than committing a directional call.
- **Recalibrate confidence** against the last 20 outcomes — the Oct upward drift is unearned; tie confidence to out-of-sample hit rate, not model internals.
- **Directional loss over magnitude:** retrain/weight selection to penalize sign errors, since 58% of fails are directional.