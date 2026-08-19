# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction is the core problem, not magnitude.** 19 of 28 fails are directional misses (68%). The model gets the size roughly right but the sign wrong — it's a coin-flip on up/down, and that's what kills pass rates.
- **Barely beating a naive baseline.** On most fails APE ≈ baseline_ape (e.g. 07-15 0.0298 vs 0.0293, 06-17 0.0373 vs 0.0358). The ensemble adds almost no edge over "predict yesterday's close."
- **Fat-tail blowups on gap days.** 07-31 APE 13.2%, plus repeated 3-5% misses (06-22, 06-25, 07-23, 07-30). These large-move sessions are systematically mispredicted in direction.
- **`macro` and `technical` are dead weight.** macro hit_rate 10.6%, technical 17%. Yet macro repeatedly wins the ensemble on fail days (06-12, 07-13, 07-15, 07-30) — the weight-selection is picking the worst strategy at the worst time.

## Unreliable under these conditions
- **High-volatility / gap sessions:** the largest APEs cluster where the daily move is big; the model damps toward small moves and misses sign.
- **When `macro` or `news` "wins":** macro-led days fail ~persistently; news wins often but its directional calls flip on volatile days (07-23, 07-31 large misses even when nominally "beats baseline").
- **Confidence is uninformative, not overconfident.** 0 overconfident misses only because confidence is uniformly low (0.1–0.6). It has no separation: passes at 0.1, fails at 0.55. Calibration is flat — confidence carries zero signal.

## Fixes to try next
- Add an explicit **direction classifier** separate from magnitude; current blend optimizes size and ignores sign.
- **Down-weight or drop macro** (0.106 hit rate) and cap technical; stop letting the selector crown low-hit-rate strategies on volatile days.
- **Regime gate:** detect high-vol/gap days (overnight move, earnings/event dates) and switch to a wider, sign-aware model or abstain.
- **Recalibrate confidence** against realized hit rate — right now it's noise; either fix it or stop reporting it.
- Benchmark hard against persistence baseline; kill any config that doesn't clear it on directional hit rate, not just APE.