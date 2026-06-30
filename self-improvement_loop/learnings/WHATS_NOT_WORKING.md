# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the killer**: 7 of 8 fails are directional misses (only 1 pure magnitude miss). The model gets the sign of the move wrong, then bleeds APE. Mean fail APE 3.36% is large for a daily AMZN close.
- **Errors compound late in the window**: the recent run (06-22 → 06-29) is 5 straight fails, all directional, with APE climbing to ~5%. Quality is decaying over time, not improving.
- **Losing to baseline on the misses**: most fails have `beats_baseline=false` — the blend is actively worse than naive on the days it matters.
- Overconfidence is technically zero only because **confidence is uniformly low (0.38–0.58)** — the model never commits, so calibration looks fine by accident. It's underconfident even on its few wins (06-08 win at 0.5).

## Unreliable under these conditions
- **momentum and macro lead into fails**: momentum "won" on 3 of the worst directional misses (06-12, 06-25); macro has the worst hit rate (8%) and highest MAPE (2.6%) — it should rarely lead.
- **news-led days are a mixed bag**: best hit rate (33%) but still drove the 5% blowup on 06-22 and the 06-29 fail. News is high-variance, not reliable.
- **contrarian is consistently down-weighted (0.05–0.16) yet still wins 2x by luck** — it's noise, not signal.
- **Choppy/turning markets**: every directional miss clusters where the prior trend likely reversed; the blend is trend-following into reversals.

## Fixes to try next
- Add a **regime/reversal filter**: when recent volatility or trend-flip risk is high, cut momentum and macro weight, lean baseline.
- **Demote macro hard** (hit rate 8%) and cap contrarian as a tiebreaker only.
- Build a dedicated **direction classifier** separate from magnitude; current blend optimizes level, fails sign.
- **Recalibrate confidence upward on agreement, downward on strategy disagreement** — current flat ~0.4 is uninformative.
- Investigate the **06-22 → 06-29 decay**: likely stale weights chasing a regime that already changed.