# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the primary failure**: 18 of 26 fails are directional misses (69%). The model gets the size roughly right but the sign wrong — it's a directional coin-flip on hard days.
- **Baseline-parity trap**: On most big-APE fails the prediction barely differs from baseline (e.g. 06-15, 06-22, 06-25, 07-15, 07-30), meaning the ensemble adds nothing when it matters — it just tracks yesterday's price and eats the move.
- **Fat-tail blowups**: 07-31 (13.2% APE) and the 3–5% cluster (06-17, 06-22, 07-23, 07-30) show the model has no defense against large single-day gaps (likely earnings/news shocks).
- **macro is dead weight**: 12% hit rate, worst MAPE, and it "wins" precisely on the worst days (07-13, 07-15, 07-30). When macro is the deciding strategy, expect a fail.

## Unreliable under these conditions
- **High-volatility / gap days**: every APE >3% is a directional miss where the ensemble hugged baseline. No regime detection.
- **macro-led and technical-led days**: macro (12% hit) and technical (14.6% hit) are near-random; both are over-weighted (~0.20 hint) relative to their skill.
- **Mid-confidence band (0.4–0.55)**: this is where the fails cluster (06-12, 06-25, 07-13, 08-07, 08-10). Not classic overconfidence (0 flagged), but confidence is flat/uninformative — it doesn't separate hits from misses. Effectively miscalibrated by being noise.
- **News-led days are the least-bad** (29% hit, best MAPE) but still fail more than half the time — no strategy is reliable standalone.

## Fixes to try next
- **Cut or down-weight macro and technical hard**; shift mass toward news (best MAPE/hit) and momentum. Current weight hints reward the worst performers.
- **Add a volatility/gap regime flag**: on high-vol days, widen intervals and stop hugging baseline — that's where directional misses concentrate.
- **Rebuild the directional model separately from magnitude** — sign is the bottleneck, not scale.
- **Recalibrate confidence**: it currently carries no signal (fails span 0.1–0.58). Tie confidence to strategy agreement and recent regime, then suppress trades in the 0.4–0.55 dead zone.
- **Earnings/event calendar overlay** to pre-empt the 07-31-style tail blowups.