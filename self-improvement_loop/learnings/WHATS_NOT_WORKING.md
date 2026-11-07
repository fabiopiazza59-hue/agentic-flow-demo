# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Overall pass rate is weak (46/104, 44%), and directional misses (31) slightly outnumber magnitude misses (27) — the model struggles with *both* direction and sizing, not just one.
- Big-tail blowups recur on event days: 2026-07-31 (13.2% APE), 2026-11-04 (6.7%), 2026-10-19 (5.5%), 2026-10-30 (5.2%), 2026-06-22 (5.1%). These dominate mean fail APE (2.45%) and cluster around earnings/macro prints.
- `macro` and `technical` are net value-destroyers: hit rates 12.5% and 11.5%, highest MAPEs (1.73%, 1.55%). When `macro` is the winning strategy, outcomes are consistently poor (06-12, 07-13, 07-15, 08-19, 10-30).
- Many "wins" are hollow — they pass APE but still lose on direction and fail to beat baseline (e.g. 08-17, 08-27, 09-02, 10-01, 10-05). Passing the threshold ≠ adding edge.

## Unreliable under these conditions
- High-volatility / large-move days (baseline APE >2.5%): model barely dents the baseline and often misses direction — these are where all the tail losses live.
- Late-Oct/early-Nov stretch (10-29 → 11-05): sustained large errors, likely earnings + macro regime it can't price.
- Overconfidence is creeping in: recent high-confidence calls (0.62–0.72) on 08-19, 09-03, 09-11, 09-09, 06-30, 07-13 were misses. Confidence rose in Oct (0.5–0.66) while pass rate stayed flat — calibration is drifting the wrong way.
- `contrarian` has decent hit rate (27%) but produces the worst tail losses (10-19, 11-04) — reliable in chop, dangerous in trends.

## Fixes to try next
- Cut or floor `macro` and `technical` weights; they lose on both accuracy and direction. Shift mass toward `news` (best hit rate 33%, lowest MAPE).
- Add an event-day guard (earnings/CPI/FOMC): widen intervals, down-weight contrarian, and suppress confidence on known high-vol dates.
- Recalibrate confidence against realized hit rate — current high-confidence buckets are no better than low ones; shrink confidence toward 0.5 until calibration validates.
- Track "beats_baseline" as the real KPI, not APE-pass; penalize hollow passes that lose direction.