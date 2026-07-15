# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not magnitude.** 11 of 13 fails are directional misses (85%); only 2 are pure magnitude. The model is guessing up/down wrong, not sizing wrong. APE on fails averages 2.55% — big enough to matter.
- **Losing to the naive baseline.** On failing days the ensemble beats baseline only occasionally; most misses (e.g. 06-12, 06-17, 06-22, 06-25, 06-26, 06-29, 07-13) have ape ≥ baseline_ape. The stack is adding negative value on the hard days.
- **The big blowups cluster.** 06-15 through 06-29 is a sustained losing streak (~3–5% APE repeatedly) — a regime the model never adapted to. That two-week stretch drives most of the total error.
- **Every strategy has a losing directional hit rate.** Best is news at 32%, worst is macro at 9%. A coin flip beats most of them. Whatever signal is being extracted, it's near-random or inverted.

## Unreliable under these conditions
- **Trending/volatile regimes (mid-to-late June):** consecutive same-direction misses suggest the model kept fading a move that continued. Contrarian (14% hit) and macro (9% hit) are actively harmful here.
- **When "news" is the winning strategy:** it wins most raw picks (7) but still fails on 06-15, 06-22, 06-25, 06-29, 07-01, 07-08 — it's overweighted (news carries ~0.28–0.42 weight on nearly every failing day) yet unreliable directionally.
- **Confidence gives no signal.** Fails span conf 0.38–0.58, passes span 0.40–0.55 — nearly identical. 0 overconfident-miss flags is misleading: confidence is flat/uninformative, not well-calibrated. Highest-conf days (0.55–0.58: 06-12, 07-13) all failed.

## Fixes to try next
- **Add a directional regime filter.** Detect trend persistence and suppress contrarian/macro (both <15% hit) when a move is running; stop fading continuations.
- **Demote macro and contrarian hard**; cap news weight — its dominance (0.3–0.4) isn't earning its keep on miss days.
- **Rebuild confidence calibration.** Current confidence is a flat band with no predictive power; recalibrate against realized hit rate or gate low-conviction days out entirely.
- **Sanity-gate against baseline:** if ensemble direction disagrees with persistence and confidence is low, default toward baseline rather than the stack.