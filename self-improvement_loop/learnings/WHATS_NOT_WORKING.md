# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Directional failure is the dominant problem**: 17 of 25 fails (68%) are direction misses, not just magnitude. The model isn't sized wrong — it's pointed wrong. This is a signal problem, not a calibration one.
- **We barely beat the naive baseline.** On failed days, mean fail APE (2.77%) roughly matches or exceeds baseline_ape repeatedly (e.g. 06-12, 06-15, 06-25, 07-13, 07-21, 07-24). We're adding noise, not edge.
- **Big-move days are catastrophic**: 07-31 (13.2% APE), 06-22 (5.1%), 07-30 (3.9%), 07-23 (4.2%). The model has no capacity to catch or contain large single-day swings — likely earnings/gap events.
- **News-weighted days go wrong most often.** When news weight is high (~0.30-0.42) the prediction frequently misses direction (06-25, 07-01, 07-13, 07-20, 07-21, 07-24). News is over-weighted (0.2269 hint) relative to its real reliability.

## Unreliable under these conditions
- **High-volatility / gap days (APE > 3%)**: every strategy fails; macro and news "win" these days but only because everything else is worse.
- **When macro is the winning strategy**: hit_rate 0.128, worst MAPE (0.0197). Macro winning is a red flag it's a leftover/default, not a real signal.
- **Technical as winner**: hit_rate 0.154 — nearly a coin-flip loser. Weighted at 0.207 but underperforms news.
- **Low-confidence regime doesn't correlate with misses cleanly** — some 0.1 confidence days pass (07-27, 08-03), some fail (07-30). Confidence carries no information.

## Fixes to try next
- **Cut news and macro weight**; both are over-weighted vs. their hit rates. Tilt toward momentum/news blend only when they agree on direction.
- **Add a direction-gate**: if strategies disagree on sign, output a smaller/near-flat move rather than committing — most damage is wrong-sign conviction.
- **Detect high-vol/event days** (earnings calendar, prior-day range) and either widen intervals or abstain; these days drive the tail losses.
- **Recalibrate confidence** — it's currently uninformative; either fix it or stop reporting it. Overconfident_misses=0 suggests confidence is uniformly low/meaningless, not well-calibrated.