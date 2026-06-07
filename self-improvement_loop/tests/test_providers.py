"""Provider fallback: keyed provider failure must fall back to keyless Stooq."""

import pandas as pd

import src.data.providers as providers


def _fake_history():
    idx = pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03"]).date
    return pd.DataFrame(
        {"open": [180, 181, 182], "high": [183, 184, 185], "low": [179, 180, 181],
         "close": [182.0, 183.0, 184.5], "volume": [1, 2, 3]},
        index=list(idx),
    )


def test_quote_falls_back_to_history(monkeypatch):
    # No keyed provider and no live quote -> derive prev_close from latest completed session.
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_fetch_history", lambda symbol: _fake_history())
    monkeypatch.setattr(providers, "_fetch_quote", lambda symbol: None)
    q = providers.get_quote("AMZN")
    assert q["source"] == "history"
    assert q["prev_close"] == 184.5
    assert q["asof"] == "2024-05-03"


def test_actual_close_lookup(monkeypatch):
    monkeypatch.setattr(providers, "_fetch_history", lambda symbol: _fake_history())
    assert providers.get_actual_close("AMZN", "2024-05-02") == 183.0
    assert providers.get_actual_close("AMZN", "2024-12-25") is None  # not a session in fixture


def test_stooq_pow_solver():
    # difficulty-1 hashcash: solver must find n with SHA256(c+n) starting with one hex zero.
    import hashlib
    n = providers._solve_stooq_pow("seed", 1)
    assert hashlib.sha256(f"seed{n}".encode()).hexdigest().startswith("0")
