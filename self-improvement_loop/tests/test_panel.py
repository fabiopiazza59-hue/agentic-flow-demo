"""Panel: batched LLM adjustments, research/refresh/score, session-level verdicts."""

import json
import math
import random
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import settings
from src.panel import analyst, evolve, report, runner

NOW = datetime(2026, 9, 27, 21, 45, tzinfo=timezone.utc)     # Sunday evening: target = Mon 28th
SYMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ", "KKK", "LLL"]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    for key, rel in (("PANEL_LEDGER_PATH", "data/panel.jsonl"),
                     ("PANEL_STATE_PATH", "learnings/panel_state.json"),
                     ("PANEL_METRICS_JSON", "results/panel_metrics.json"),
                     ("PANEL_RESULTS_MD", "RESULTS_PANEL.md"),
                     ("PROMPT_REGISTRY_PATH", "learnings/prompt_variants.json")):
        monkeypatch.setattr(settings, key, tmp_path / rel, raising=False)
    monkeypatch.setattr(settings, "PANEL_SYMBOLS", tuple(SYMS), raising=False)
    return tmp_path


def _hist(price=100.0, n=60, seed=0):
    rng = random.Random(seed)
    closes = [price]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.gauss(0, 0.015)))
    idx = list(pd.bdate_range(end="2026-09-25", periods=n).date)
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                         "volume": [1] * n}, index=idx)


def _fake_market(monkeypatch, gap=0.01):
    hists = {s: _hist(seed=i) for i, s in enumerate(SYMS)}
    monkeypatch.setattr(runner.data, "daily_history", lambda symbols, through, **k:
                        {s: hists[s] for s in symbols if s in hists})
    monkeypatch.setattr(runner.data, "extended_last", lambda symbols, since, until: {
        s: {"price": float(hists[s]["close"].iloc[-1]) * (1 + gap), "live": True,
            "time": "2026-09-25T23:00:00+00:00", "source": "yfinance_ext"} for s in symbols})
    return hists


class _Client:
    """Minimal stand-in for the Anthropic client: returns canned text blocks."""

    def __init__(self, payload):
        self.payload, self.messages = payload, self

    def create(self, **kwargs):
        blocks = [SimpleNamespace(type="web_search_tool_result", content=[1, 2]),
                  SimpleNamespace(type="text", text=json.dumps(self.payload))]
        return SimpleNamespace(content=blocks)


# ---------------------------------------------------------------------------------- analyst

def test_offline_adjustments_are_zero():
    out = analyst.adjustments({"AAA": {}}, "2026-09-28", "t", client=None)
    assert out["AAA"]["sigma"] == 0.0


def test_adjustments_are_parsed_clamped_and_default_to_zero(monkeypatch):
    names = {s: {"prev_close": 100.0, "anchor": 101.0, "anchor_live": True, "gap": 0.01,
                 "sigma": 0.02} for s in ("AAA", "BBB", "CCC")}
    client = _Client({"adjustments": {"AAA": {"sigma": 3.0, "reason": "earnings"},
                                      "BBB": {"sigma": -0.2, "reason": "x"}}})
    out = analyst.adjustments(names, "2026-09-28", "t", client)
    assert out["AAA"]["raw"] == 3.0 and out["AAA"]["sigma"] == settings.PANEL_MAX_ADJ_SIGMA
    assert out["BBB"]["sigma"] == -0.2 and out["CCC"]["sigma"] == 0.0
    assert out["AAA"]["web_results"] == 2


# ----------------------------------------------------------------------------------- runner

def test_research_logs_the_llm_counterfactual_and_respects_the_kill_switch(monkeypatch, isolated):
    _fake_market(monkeypatch)
    client = _Client({"adjustments": {"AAA": {"sigma": 0.5, "reason": "x"}}})
    rows = runner.research([], pd.Timestamp("2026-09-28").date(), client, NOW)
    assert len(rows) == len(SYMS)
    a = next(r for r in rows if r["symbol"] == "AAA")
    assert a["anchor_live"] and a["predicted_close_llm"] > a["anchor"]
    assert a["predicted_close"] == a["predicted_close_llm"] and a["late_minutes"] < 0

    runner.save_state({"llm_enabled": False})
    rows = runner.research([], pd.Timestamp("2026-09-28").date(), client, NOW)
    a = next(r for r in rows if r["symbol"] == "AAA")
    assert a["predicted_close"] == a["anchor"]                 # ships the free baseline
    assert a["predicted_close_llm"] > a["anchor"]              # ...but keeps measuring the LLM


def test_research_refuses_after_the_cutoff(monkeypatch, isolated):
    _fake_market(monkeypatch)
    late = datetime(2026, 9, 28, 13, 25, tzinfo=timezone.utc)
    assert runner.research([], pd.Timestamp("2026-09-28").date(), None, late) == []


def test_refresh_and_score(monkeypatch, isolated):
    hists = _fake_market(monkeypatch, gap=0.0)
    target = pd.Timestamp("2026-09-28").date()
    rows = runner.research([], target, None, NOW)
    # a fresher pre-market trade 2% up
    monkeypatch.setattr(runner.data, "extended_last", lambda symbols, since, until: {
        s: {"price": float(hists[s]["close"].iloc[-1]) * 1.02, "live": True,
            "time": "2026-09-28T12:00:00+00:00", "source": "yfinance_ext"} for s in symbols})
    rows = runner.refresh(rows, target, datetime(2026, 9, 28, 12, 5, tzinfo=timezone.utc))
    assert all(r["anchor_updates"] == 1 for r in rows)
    # score after the close
    monkeypatch.setattr(runner.data, "daily_history", lambda symbols, through, **k: {
        s: pd.DataFrame({"close": [float(hists[s]["close"].iloc[-1]) * 1.03]}, index=[target])
        for s in symbols})
    rows = runner.score(rows, datetime(2026, 9, 28, 21, 0, tzinfo=timezone.utc))
    assert all(r["status"] == "scored" and r["llm_ape"] is not None for r in rows)
    assert all(r["anchor_ape"] < r["baseline_ape"] for r in rows)   # the fresh anchor was closer


# ----------------------------------------------------------------------------------- report

def _simulated(adjuster, sessions=15, seed=3):
    """Scored panel rows where the realized move off the anchor is known to the adjuster."""
    rng, rows = random.Random(seed), []
    dates = [d.isoformat() for d in pd.bdate_range("2026-06-01", periods=sessions).date]
    for d in dates:
        for s in SYMS:
            prior, sigma = 100.0, 0.02
            anchor = prior * math.exp(rng.gauss(0, 0.005))
            move = rng.gauss(0, sigma)
            actual = anchor * math.exp(move)
            adj = adjuster(move / sigma, rng)
            llm = anchor * (1 + adj * sigma)
            rows.append({"date": d, "symbol": s, "status": "scored",
                         "created_at": f"{d}T01:00:00+00:00", "prior_close": prior,
                         "anchor": anchor, "sigma_pct": sigma, "adj_sigma": adj,
                         "predicted_close_llm": llm, "predicted_close": llm,
                         "actual_close": actual, "llm_ape": abs(llm - actual) / actual,
                         "anchor_ape": abs(anchor - actual) / actual,
                         "baseline_ape": abs(prior - actual) / actual})
    return rows


def test_report_detects_skill_noise_and_harm():
    skilled = report.build(_simulated(lambda z, r: max(-0.5, min(0.5, 0.5 * z))))
    assert skilled["cs"]["ape_gain_vs_anchor"]["decision"] == "positive"
    assert skilled["cs"]["rank_ic"]["decision"] == "positive"
    assert skilled["state"]["llm_enabled"] is True

    noise = report.build(_simulated(lambda z, r: r.choice([-0.1, 0.0, 0.1])))
    assert noise["cs"]["ape_gain_vs_anchor"]["decision"] in ("undecided", "negative")

    harmful = report.build(_simulated(lambda z, r: -max(-0.5, min(0.5, 0.5 * z))))
    assert harmful["cs"]["ape_gain_vs_anchor"]["decision"] == "negative"
    assert harmful["state"]["llm_enabled"] is False
    assert "switched off" in report.render(harmful)


def test_late_rows_never_count():
    rows = _simulated(lambda z, r: 0.5)
    for r in rows:
        r["created_at"] = f"{r['date']}T15:00:00+00:00"          # after the 13:30 UTC open
    assert report.build(rows)["n_sessions"] == 0


def test_spearman_handles_ties_and_degenerate_input():
    assert report.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert report.spearman([0, 0, 0, 1], [1, 2, 3, 4]) == pytest.approx(0.7745966, rel=1e-6)
    assert report.spearman([0, 0, 0], [1, 2, 3]) is None


def test_generate_writes_artifacts(isolated):
    runner.save_rows(_simulated(lambda z, r: 0.0, sessions=12))
    m = report.generate()
    assert settings.PANEL_RESULTS_MD.exists() and settings.PANEL_METRICS_JSON.exists()
    assert json.loads(settings.PANEL_STATE_PATH.read_text())["n_sessions"] == m["n_sessions"] == 12


def test_challenger_prompts_run_in_shadow_and_are_scored(monkeypatch, isolated):
    hists = _fake_market(monkeypatch, gap=0.0)
    reg = evolve.load_registry()
    reg["n_created"] = 1
    reg["variants"].append({"id": "p1", "k": 1, "parent": "p0", "status": "challenger",
                            "strategy": "CHALLENGER STRATEGY " * 20, "created": "t"})
    evolve.save_registry(reg)

    class Client:
        messages = None

        def __init__(self):
            self.messages = self

        def create(self, **kw):   # the champion stays at 0; the challenger moves every name
            sigma = 0.3 if kw["system"].startswith("CHALLENGER") else 0.0
            tickers = [line.split(" | ")[0] for line in kw["messages"][0]["content"].splitlines()
                       if " | " in line and not line.startswith("ticker")]
            payload = {"adjustments": {t: {"sigma": sigma, "reason": "r"} for t in tickers}}
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    target = pd.Timestamp("2026-09-28").date()
    rows = runner.research([], target, Client(), NOW)
    assert all(r["prompt_variant"] == "p0" and r["adj_sigma"] == 0.0 for r in rows)
    assert all(r["shadow_adj"] == {"p1": 0.3} for r in rows)       # logged, never shipped
    monkeypatch.setattr(runner.data, "daily_history", lambda symbols, through, **k: {
        s: pd.DataFrame({"close": [float(hists[s]["close"].iloc[-1]) * 1.01]}, index=[target])
        for s in symbols})
    rows = runner.score(rows, datetime(2026, 9, 28, 21, 0, tzinfo=timezone.utc))
    assert all(r["shadow_ape"]["p1"] < r["llm_ape"] for r in rows)   # the move up was right
