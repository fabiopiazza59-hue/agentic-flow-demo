"""Per-strategy scorecards — the self-improvement signal.

Each analyst's historical accuracy is tracked so the meta-judge can up-weight strategies that
have actually been working. `weight_hints` turns recent accuracy into a soft prior in [0,1].
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import settings
from .metrics import ape


def load_scorecards(path: str | Path | None = None) -> dict:
    path = Path(path or settings.SCORECARDS_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_scorecards(scorecards: dict, path: str | Path | None = None) -> None:
    path = Path(path or settings.SCORECARDS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scorecards, indent=2, ensure_ascii=False), encoding="utf-8")


def _blank() -> dict:
    return {"n": 0, "wins": 0, "sum_ape": 0.0, "hit_rate": 0.0, "mape": 0.0, "weight_hint": 0.2}


def update_scorecards(scorecards: dict, analyst_predictions: dict, actual: float) -> dict:
    """Update each strategy's stats given its prediction and the realized close.

    `analyst_predictions` maps strategy -> {"predicted_close": float, ...}.
    The strategy with the smallest ape for this day gets a "win".
    Returns the mutated scorecards dict.
    """
    if not analyst_predictions:
        return scorecards

    apes: dict[str, float] = {}
    for name, pred in analyst_predictions.items():
        p = pred.get("predicted_close")
        if p is None:
            continue
        apes[name] = ape(float(p), actual)

    if not apes:
        return scorecards

    winner = min(apes, key=apes.get)
    for name, a in apes.items():
        card = scorecards.get(name) or _blank()
        card["n"] += 1
        card["sum_ape"] += a
        if name == winner:
            card["wins"] += 1
        card["hit_rate"] = card["wins"] / card["n"]
        card["mape"] = card["sum_ape"] / card["n"]
        scorecards[name] = card

    return weight_hints(scorecards)


def weight_hints(scorecards: dict) -> dict:
    """Derive a soft weight prior per strategy from inverse-MAPE, normalized to sum≈1.

    Strategies with fewer than 3 observations keep a neutral prior so the judge isn't misled early.
    """
    eligible = {n: c for n, c in scorecards.items() if c.get("n", 0) >= 3 and c.get("mape", 0) > 0}
    if eligible:
        inv = {n: 1.0 / c["mape"] for n, c in eligible.items()}
        total = sum(inv.values())
        for n, c in scorecards.items():
            c["weight_hint"] = round(inv[n] / total, 4) if n in inv else round(0.5 / max(total, 1), 4)
    else:
        for c in scorecards.values():
            c["weight_hint"] = 0.2
    return scorecards
