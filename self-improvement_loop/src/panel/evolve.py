"""Prompt evolution on the panel — variants proposed by reflection, promoted only on proof.

The loop may now change its own instructions, but only the way the lab changes its mechanisms:

1. Variants. The analyst's system prompt is split into an evolvable *strategy* (text) and a
   fixed *contract* in code (tool budget, the ±σ cap, the JSON format) that no variant can
   touch — so a variant can change what the model thinks about, never what the system accepts.
2. Shadow runs. Each session the champion's adjustments ship; up to EVOLVE_MAX_CHALLENGERS
   challengers run on the same snapshot and are logged per row (`shadow_adj`), costing one
   extra batched call set each and changing nothing that ships.
3. Promotion. A challenger replaces the champion only when an anytime-valid confidence sequence
   on the session-averaged APE gain (champion APE − challenger APE) excludes zero. Because the
   loop can propose challengers forever, the error budget is spent across all of them:
   the k-th challenger ever created is tested at alpha / (k(k+1)), which sums to alpha — so the
   chance that *any* worse-or-equal prompt is ever promoted stays below alpha.
   A challenger that is shown worse, or is still undecided after EVOLVE_MAX_SESSIONS, retires.
4. Proposals (reflective mutation, in the spirit of GEPA-style prompt evolution). When a slot is
   free, the reflector model reads the champion's strategy, its measured failure cases and the
   record of every variant tried so far, and writes one new strategy. Its input is numbers only
   — never the analysts' free-text reasons, which were shaped by web pages and could otherwise
   carry injected instructions into future prompts.

Everything lives in learnings/prompt_variants.json, so every prompt that ever ran is in git.
"""

from __future__ import annotations

import json
import math
import statistics as st

from ..config import settings
from ..evals import integrity
from ..evals.metrics import ape
from ..evals.sequential import confidence_sequence
from ..utils import extract_json, now_iso
from .analyst import SEED_STRATEGY


def log(msg: str) -> None:
    print(f"[evolve] {msg}", flush=True)


# ------------------------------------------------------------------------------- registry

def load_registry() -> dict:
    p = settings.PROMPT_REGISTRY_PATH
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"n_created": 0, "variants": [{
        "id": "p0", "k": 0, "parent": None, "status": "champion", "created": None,
        "strategy": SEED_STRATEGY, "rationale": "seed: the hand-written panel prompt"}]}


def save_registry(reg: dict) -> None:
    p = settings.PROMPT_REGISTRY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def champion(reg: dict) -> dict:
    return next(v for v in reg["variants"] if v["status"] == "champion")


def challengers(reg: dict) -> list[dict]:
    return [v for v in reg["variants"] if v["status"] == "challenger"]


def alpha_for(k: int) -> float:
    """Error budget for the k-th challenger ever created (k >= 1): sums to LAB_ALPHA."""
    return settings.LAB_ALPHA / (k * (k + 1))


# ----------------------------------------------------------------------------- evaluation

def _clean_scored(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "scored" and not integrity.is_late(r)
            and r.get("llm_ape") is not None]


def challenger_series(rows: list[dict], vid: str) -> list[float]:
    """Per session: mean over names of (champion APE − challenger APE); positive = better."""
    by_day: dict[str, list[float]] = {}
    for r in _clean_scored(rows):
        sa = (r.get("shadow_ape") or {}).get(vid)
        if sa is not None:
            by_day.setdefault(r["date"], []).append(float(r["llm_ape"]) - float(sa))
    return [st.mean(v) for _, v in sorted(by_day.items())]


def evaluate(rows: list[dict], reg: dict) -> tuple[dict, list[str]]:
    """Update challenger stats; promote at most one proven winner; retire the rest as due."""
    events: list[str] = []
    for v in challengers(reg):
        cs = confidence_sequence(challenger_series(rows, v["id"]), alpha=alpha_for(v["k"]))
        v["stats"] = {k: cs[k] for k in ("n", "mean", "lo", "hi", "decision", "alpha")}
    winners = [v for v in challengers(reg) if v["stats"]["decision"] == "positive"]
    if winners:
        best = max(winners, key=lambda v: v["stats"]["mean"])
        old = champion(reg)
        old.update(status="retired", retired=now_iso(), why=f"superseded by {best['id']}")
        best.update(status="champion", promoted=now_iso())
        events.append(f"promoted {best['id']} over {old['id']} "
                      f"(gain {best['stats']['mean'] * 100:+.3f}%/session, n={best['stats']['n']})")
    for v in challengers(reg):
        st_ = v["stats"]
        if st_["decision"] == "negative":
            v.update(status="retired", retired=now_iso(), why="worse than the champion")
            events.append(f"retired {v['id']}: worse")
        elif st_["n"] >= settings.EVOLVE_MAX_SESSIONS:
            v.update(status="retired", retired=now_iso(), why="undecided after max sessions")
            events.append(f"retired {v['id']}: inconclusive after {st_['n']} sessions")
    return reg, events


# ------------------------------------------------------------------------------ proposals

_PROPOSER = (
    "You improve the strategy text of an LLM equity analyst. Each session the analyst sees a "
    "panel of US large caps, each with a free baseline (its latest pre-open price, the anchor), "
    "and outputs for every name a move off the anchor in units of that name's daily σ, clamped "
    "to ±{cap}σ by code. It is scored by whether its forecasts beat the anchor. The output "
    "format, tool budget and clamp are fixed by the system; you may only rewrite the STRATEGY: "
    "what the analyst should look for, when to move a name and when to stay at 0.\n"
    "You get the current strategy, measured failure cases (numbers only) and every variant tried "
    "so far with its measured result. Propose ONE new strategy that addresses the failure "
    "pattern and differs meaningfully from variants that already failed. Keep it under "
    "{chars} characters and do not describe output formats.\n"
    "Respond with ONLY JSON: {{\"strategy\": \"<text>\", \"rationale\": \"<2 sentences>\"}}"
)


def failure_cases(rows: list[dict], vid: str, n: int = 8) -> dict:
    """Numbers-only evidence: where the champion hurt most, and the big moves it missed."""
    mine = [r for r in _clean_scored(rows) if r.get("prompt_variant", "p0") == vid]
    fields = lambda r: {  # noqa: E731
        "symbol": r["symbol"], "date": r["date"], "adj_sigma": r.get("adj_sigma"),
        "gap_pct": round((float(r["anchor"]) / float(r["prior_close"]) - 1) * 100, 2),
        "realized_off_anchor_sigma": round(
            math.log(float(r["actual_close"]) / float(r["anchor"])) / float(r["sigma_pct"]), 2),
        "llm_minus_anchor_ape_pct": round((r["llm_ape"] - r["anchor_ape"]) * 100, 3)}
    hurt = sorted(mine, key=lambda r: r["llm_ape"] - r["anchor_ape"], reverse=True)[:n]
    missed = sorted(mine, key=lambda r: abs(math.log(float(r["actual_close"]) / float(r["anchor"]))
                                            / float(r["sigma_pct"])), reverse=True)[:n]
    moved = sum(1 for r in mine if r.get("adj_sigma"))
    return {"n_forecasts": len(mine), "share_moved": round(moved / len(mine), 3) if mine else 0,
            "mean_gain_vs_anchor_pct": round(st.mean(r["anchor_ape"] - r["llm_ape"] for r in mine)
                                             * 100, 4) if mine else None,
            "hurt_most": [fields(r) for r in hurt], "largest_moves": [fields(r) for r in missed]}


def _history(reg: dict) -> list[dict]:
    return [{"id": v["id"], "status": v["status"], "why": v.get("why"),
             "stats": v.get("stats"), "strategy_excerpt": v["strategy"][:400]}
            for v in reg["variants"]]


def propose(rows: list[dict], reg: dict, client) -> tuple[dict, str | None]:
    """Add one reflective challenger if a slot is free and the champion has enough history."""
    if client is None or len(challengers(reg)) >= settings.EVOLVE_MAX_CHALLENGERS:
        return reg, None
    champ = champion(reg)
    sessions = {r["date"] for r in _clean_scored(rows) if r.get("prompt_variant", "p0") == champ["id"]}
    if len(sessions) < settings.EVOLVE_MIN_SESSIONS:
        return reg, None
    user = json.dumps({"current_strategy": champ["strategy"],
                       "failures": failure_cases(rows, champ["id"]),
                       "variants_tried": _history(reg)}, indent=2)
    try:
        resp = client.messages.create(
            model=settings.REFLECTOR_MODEL, max_tokens=settings.REFLECTOR_MAX_TOKENS * 2,
            system=_PROPOSER.format(cap=settings.PANEL_MAX_ADJ_SIGMA,
                                    chars=settings.EVOLVE_MAX_STRATEGY_CHARS),
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        raw = extract_json(text)
    except Exception:  # noqa: BLE001 - no proposal this run
        return reg, None
    strategy = str(raw.get("strategy", "")).strip()
    if not (200 <= len(strategy) <= settings.EVOLVE_MAX_STRATEGY_CHARS) or \
            any(strategy == v["strategy"] for v in reg["variants"]):
        return reg, None
    reg["n_created"] = int(reg.get("n_created", 0)) + 1
    vid = f"p{reg['n_created']}"
    reg["variants"].append({"id": vid, "k": reg["n_created"], "parent": champ["id"],
                            "status": "challenger", "created": now_iso(), "strategy": strategy,
                            "rationale": str(raw.get("rationale", ""))[:600]})
    return reg, vid


def step(rows: list[dict], client=None) -> dict:
    """Evaluate, promote/retire, maybe propose; persist and return the registry."""
    reg = load_registry()
    reg, events = evaluate(rows, reg)
    reg, new = propose(rows, reg, client)
    if new:
        events.append(f"proposed {new} (tested at alpha={alpha_for(reg['n_created']):.4f})")
    for e in events:
        log(e)
    save_registry(reg)
    return reg


def shadow_ape(row: dict, actual: float) -> dict:
    """APE each shadow variant would have scored (anchor-relative, so re-anchoring carries over)."""
    anchor, sigma = float(row["anchor"]), float(row["sigma_pct"])
    return {vid: ape(anchor * (1.0 + float(a) * sigma), actual)
            for vid, a in (row.get("shadow_adj") or {}).items()}
