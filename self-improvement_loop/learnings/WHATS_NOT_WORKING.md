# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional misses dominate failures (27 of 43, ~63%).** The system is picking the wrong *sign* more than it's sizing moves wrong — this is a signal problem, not a calibration-of-magnitude problem.
- **The ensemble barely beats baseline.** A large share of failures also fail `beats_baseline`, meaning on hard days the blend adds nothing over a naive hold/persistence call.
- **Tail blowups go uncaught.** The 2026-07-31 miss (APE ~13%, likely earnings) and repeated 3-5% APE days show no event/vol guardrail — the model keeps committing on gap days.
- **macro is the worst winner.** When `macro` is the winning strategy it fails often (2026-07-13, -15, -19, -30); hit_rate 0.11, highest MAPE (0.0163). It's actively dragging the blend.
- **technical is nearly a coin-flip-loser:** hit_rate 0.136, yet still carries the 2nd-highest weight hint (0.208). Weight does not track reliability.

## Unreliable under these conditions
- **High-volatility / large-move days (APE >3%):** nearly all big-APE entries are directional misses — the model underperforms exactly when moves are big.
- **When macro or technical is the lead strategy:** combined these drive a disproportionate share of fails.
- **Recent high-confidence regime (Sept–Oct, conf 0.52–0.66):** passes on small-move days but still misses direction on several (09-11, 09-17, 09-22, 10-06). Confidence rose system-wide without accuracy rising — **miscalibration is creeping in**: conf 0.72 on 08-31 and 09-09 both failed.
- **Overconfident misses (5):** small count but concentrated at conf ≥0.54 (08-19 @0.62, 08-20 @0.54, 09-03 @0.62) — the model is most wrong when it feels most sure on choppy days.

## Fixes to try next
- **Cut macro's weight toward ~0 and cap technical** until hit_rate recovers; reallocate to news (best: 0.333 hit, lowest MAPE).
- **Recalibrate confidence against realized hit-rate** — current conf 0.5–0.66 band shows no edge; shrink confidence on high-vol days.
- **Add an event/earnings + volatility gate** that widens intervals or abstains instead of committing on gap days.
- **Weight strategies by trailing directional hit-rate, not static hints** — current weights invert the reliability ranking.
- **Track directional accuracy as the primary KPI**, not APE, since sign errors are the core failure.