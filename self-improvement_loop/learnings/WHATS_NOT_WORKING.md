# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate**: 27 of 42 fails (64%) are wrong-direction, not just magnitude. This is a signal problem, not a calibration-of-size problem. The model can't call the sign on choppy days.
- **We barely beat baseline**: many fails have ape ≈ baseline_ape (e.g. 2026-07-15, 08-20, 09-28). On hard days we add ~zero edge over naive persistence.
- **macro is dead weight**: 11.8% hit rate, worst MAPE (0.0167), yet still carries ~18% weight. Every day macro "wins" the routing (06-12 setup, 07-13, 07-15, 07-30, 08-19) it fails. macro-led days are near-automatic misses.
- **technical over-weighted vs. performance**: 14.5% hit rate but 0.209 weight_hint (2nd highest). It earns its worst outcomes on the days it's routed to lead.
- **Large-error blowups cluster**: 06-22, 07-23, 07-31 (13% ape!), 07-30 — big moves the ensemble completely fails to size or direct.

## Unreliable under these conditions
- **High-volatility / gap days**: whenever the true move is large (ape >3%), directional_hit collapses. The model compresses toward small moves and gets run over (07-31, 06-22, 08-19).
- **macro- and technical-led routing**: combined these two "winning_strategy" days are where most fails concentrate.
- **Mild-but-real confidence miscalibration late**: 08-31 (conf 0.72, fail), 09-09 (0.72, fail), 08-19 (0.62, fail), 09-03 (0.62, fail). Confidence has drifted up recently without accuracy following — overconfident_misses is only 4 flagged but the 0.6+ fails are growing.
- **news is the only reliable leg** (32.9% hit, best MAPE) — everything else is coin-flip-or-worse.

## Fixes to try next
- **Cut macro to near-zero weight** or gate it to confirmed macro-event days only; it's actively hurting.
- **Rebalance toward news** (raise), trim technical/macro to match their hit rates, not their historical priors.
- **Add a volatility regime detector**: on high-expected-move days, widen predicted magnitude and de-trust mean-reverting legs (technical/contrarian).
- **Recalibrate confidence**: cap confidence on macro/technical-led days; require news agreement before emitting conf >0.6.
- **Track sign-accuracy separately** as the primary KPI — magnitude tuning is secondary while 64% of fails are directional.