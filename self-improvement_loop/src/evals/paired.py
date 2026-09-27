"""Paired, day-level statistics — the inferential unit is the market day, never the trade.

Canon rule 1 of arXiv:2609.05663: "The market day is the inferential unit, never the trade.
Positions opened the same day share the tape; position-level intervals here run ~2.5x too
narrow." Every prediction in this loop is one market day, so pairing over days is the whole
requirement — these helpers just make it the default way any two arms get compared.
"""

from __future__ import annotations

import random
import statistics as st
from math import comb


def sign_test_p(wins: int, n: int) -> float | None:
    """Two-sided exact binomial sign test under H0: P(win)=0.5 (ties excluded)."""
    if n == 0:
        return None
    k = max(wins, n - wins)
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n * 2
    return round(min(1.0, p), 4)


def bootstrap_ci(deltas: list[float], draws: int = 5000, seed: int = 7,
                 level: float = 0.90) -> tuple[float, float] | tuple[None, None]:
    """Percentile bootstrap over days for the mean of paired deltas."""
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(st.mean([deltas[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    tail = (1.0 - level) / 2
    return means[int(tail * draws)], means[min(draws - 1, int((1 - tail) * draws))]


def paired_effect(deltas: list[float], draws: int = 5000) -> dict:
    """Summarize paired day deltas: mean, bootstrap CI, win count, sign-test p.

    A positive delta means the treatment helped. `significant` is True only when the interval
    excludes zero AND the sign test agrees — the paper's FIRM bar.
    """
    if not deltas:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "wins": 0, "n_decisive": 0, "sign_test_p": None, "significant": False}
    decisive = [d for d in deltas if d != 0]
    wins = sum(1 for d in decisive if d > 0)
    p = sign_test_p(wins, len(decisive))
    lo, hi = bootstrap_ci(deltas, draws=draws)
    excludes_zero = lo is not None and (lo > 0 or hi < 0)
    return {
        "n": len(deltas),
        "mean": round(st.mean(deltas), 8),
        "ci_low": round(lo, 8) if lo is not None else None,
        "ci_high": round(hi, 8) if hi is not None else None,
        "wins": wins,
        "n_decisive": len(decisive),
        "sign_test_p": p,
        "significant": bool(excludes_zero and p is not None and p < 0.05),
    }
