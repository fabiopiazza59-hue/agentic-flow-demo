# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- **Direction, not size, is the killer.** 10 of 12 fails are directional misses (only 2 pure magnitude). The model gets the sign wrong far more than it overshoots — this is a *turn/reversal* problem, not a scaling one.
- **Systematically worse than baseline on fails.** Mean fail APE 2.64% vs baseline that's often lower; on most losing days we'd have been better off just predicting the naive baseline. We are adding negative value on hard days.
- **A clustered bad streak.** 2026-06-12 through 07-01 is almost all fails (only 06-23/06-24 pass). Something regime-specific broke and the ensemble never adapted.
- **Big misses are large.** 06-22 APE 5.1%, 06-15/06-17/06-25/06-26/06-29 all >3%. These aren't near-misses; the model is confidently on the wrong side.

## Unreliable under these conditions
- **"news" as winning strategy = red flag.** It "wins" most often (7) yet presided over the worst run (06-15, 06-22, 06-25→06-01, 07-08, 07-09 all fails). It wins the internal contest but loses live — likely overfit to headlines during choppy tape.
- **Contrarian and macro are dead weight:** hit rates 10.5% and 5.3%. Contrarian fires exactly when momentum continues; macro adds nothing.
- **Momentum-heavy weightings whipsaw in reversals** (06-12, 06-25, 07-08 fails carried high momentum/technical weight).
- **Confidence is flat and useless, not overconfident.** 0 overconfident-misses only because confidence is pinned ~0.40–0.55 regardless of outcome. No calibration signal at all — passes and fails look identical.

## Fixes to try next
- **Cut macro and contrarian weight to near-zero;** they're negative-value. Reallocate cautiously — but note news' live reliability is worse than its win count suggests.
- **Add a reversal/regime detector:** when recent realized direction flips, damp momentum/technical and shrink toward baseline instead of committing to a side.
- **Fall back to baseline when ensemble disagreement is high** — we're losing to naive on exactly the days we're most confident in a call.
- **Rebuild confidence so it actually varies with expected error;** current output is a constant and cannot flag risky days.