# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude, is the core problem.** 9 of 10 fails are directional misses; only 1 is magnitude-only. The model is systematically calling the wrong sign, not just over/undershooting size.
- **We lose to the naive baseline on the misses.** On failing days APE (2.94%) runs above baseline; the ensemble adds noise rather than edge when it's wrong.
- **A sustained losing streak mid-to-late June (06-12 → 06-01).** ~9 consecutive fails, several with 3–5% APE. This is a regime the model never adapted to, not scattered bad luck.
- **No pattern of overconfidence** (0 overconfident misses) — but that's only because confidence is uniformly flat (~0.4–0.58). Confidence carries essentially no information; it's not calibrated, it's just muted.

## Unreliable under these conditions
- **Trending/volatile stretches (the big-APE days: 06-22 at 5.1%, 06-17 at 3.7%, 06-25/26/29 at ~3.2%)** — the ensemble keeps fading the move and gets the direction wrong repeatedly.
- **`contrarian` is the worst offender**: 11.8% hit rate, high APE. It's actively harmful in trending regimes yet still carries ~0.19 weight.
- **`macro` is dead weight**: 5.9% hit rate, highest MAPE. One lucky win (06-08) inflates its reputation.
- **`news`-led days are inconsistent**: wins some (06-23, 06-24) but drives several of the worst fails (06-15, 06-22, 06-29). High variance, no reliability filter.
- **Momentum/technical** are the only marginally-useful strategies and they carry the recovery (07-02 → 07-07 all passes were momentum/technical-led).

## Fixes to try next
- **Cut or heavily down-weight `contrarian` and `macro`** — combined ~0.37 weight for ~6–12% hit rates. Reallocate to momentum/technical.
- **Add a trend/volatility regime detector**: in strong trends, suppress contrarian entirely and lean momentum; the streak of directional misses is a fade-the-trend failure.
- **Rebuild confidence to be calibrated** — it's currently flat and useless. Tie it to strategy agreement and recent hit rate so we can size/skip low-conviction days.
- **Add a directional guardrail**: gate final sign on momentum+technical agreement before trusting news/contrarian sign flips.