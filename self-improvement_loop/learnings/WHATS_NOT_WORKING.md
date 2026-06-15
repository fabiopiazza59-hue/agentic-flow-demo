# What's Not Working — AMZN Close Predictor

_Auto-generated each scoring run._

## What keeps going wrong
- Sample is tiny (n=2, 1 fail) — treat everything below as hypotheses, not verdicts.
- The only failure (2026-06-12) was a **directional miss**, not a magnitude blowup: model called the wrong way and ate a 2.01% APE, also losing to baseline (1.24%).
- The miss came on the day the blend tilted heavily to **macro (0.34)** while down-weighting momentum (0.10) — yet **momentum was the winning strategy** that day. The weighting moved away from what worked.
- The clean pass (06-08) used flat equal weights (0.2 each) and crushed it (APE 0.07%). The active re-weighting on 06-12 coincided with the failure.
- Confidence was slightly higher on the miss (0.58 vs 0.50) but not flagged overconfident — calibration looks roughly honest so far.

## Unreliable under these conditions
- **technical, contrarian, news: 0/2 hit rate** and the highest MAPEs (1.28%, 1.56%, 1.35%). Currently they are dead weight.
- **Conviction re-weighting toward macro** appears to misfire on directional turns — the one time the model deviated from equal weights, it picked the wrong leader.
- Directional accuracy is the weak axis (1 of 1 misses was directional); magnitude has been fine when direction is right.

## Fixes to try next
- Stop trusting the dynamic weighting until n grows; revert to equal weights or cap any single strategy's weight (e.g. ≤0.25) — the flat blend is the only thing that's worked.
- Track momentum vs macro as a regime pair: 06-12 says momentum should not be the lowest weight when it's the day's winner.
- Down-weight or shelve technical/contrarian/news pending evidence; they have contributed zero wins and the most error.
- Add an explicit direction-confidence gate: when strategies disagree on sign, lower confidence rather than committing.
- Above all, **gather more scored days** — no pattern is statistically real at n=2.