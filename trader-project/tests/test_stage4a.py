"""Unit tests for Stage 4a — deterministic risk math.

Same inputs → identical outputs. No LLM dependency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import AccountConfig
from src.models.schemas import Direction, Horizon
from src.stages.stage4a_risk_math import compute_risk_math, parse_levels_from_scenario


def test_long_basic_sizing():
    """Basic long trade: entry $100, stop $90, target $125."""
    account = AccountConfig(total_capital_usd=2000)
    result = compute_risk_math(
        entry=100.0,
        stop=90.0,
        target=125.0,
        direction=Direction.long,
        current_price=100.0,
        proposed_horizon=Horizon.weeks,
        account=account,
    )
    # Stop distance = 10%, so max position = $40 / 0.10 = $400
    assert result.stop_distance_pct == 0.1
    assert result.target_distance_pct == 0.25
    assert result.r_multiple == 2.5
    assert result.position_size_usd == 400.0
    assert result.max_loss_usd == 40.0
    assert result.passes_risk_floor is True


def test_short_basic_sizing():
    """Basic short trade: entry $50, stop $55, target $40."""
    account = AccountConfig(total_capital_usd=2000)
    result = compute_risk_math(
        entry=50.0,
        stop=55.0,
        target=40.0,
        direction=Direction.short,
        current_price=50.0,
        proposed_horizon=Horizon.weeks,
        account=account,
    )
    assert result.stop_distance_pct == 0.1
    assert result.target_distance_pct == 0.2
    assert result.r_multiple == 2.0
    assert result.passes_risk_floor is True


def test_r_multiple_floor_failure():
    """R-multiple below 2.0 should fail risk floor."""
    account = AccountConfig(total_capital_usd=2000, r_multiple_floor=2.0)
    result = compute_risk_math(
        entry=100.0,
        stop=90.0,
        target=115.0,  # R = 1.5
        direction=Direction.long,
        current_price=100.0,
        proposed_horizon=Horizon.weeks,
        account=account,
    )
    assert result.r_multiple == 1.5
    assert result.passes_risk_floor is False
    assert "r_multiple_below_2.0" in result.risk_floor_failures


def test_cfd_breakeven():
    """CFD financing should compute breakeven days."""
    account = AccountConfig(total_capital_usd=2000, cfd_annual_financing_rate=0.07)
    result = compute_risk_math(
        entry=100.0,
        stop=90.0,
        target=125.0,
        direction=Direction.long,
        current_price=100.0,
        proposed_horizon=Horizon.weeks,
        account=account,
    )
    # Daily cost = 400 * 0.07 / 365 ≈ 0.0767
    # Target profit = 400 * 0.25 = 100
    # Breakeven = 100 / 0.0767 ≈ 1303 days
    assert result.cfd_breakeven_days > 100
    assert result.cfd_overnight_cost_per_day > 0


def test_parse_levels_long():
    """Extract price levels from scenario text."""
    entry, stop, target = parse_levels_from_scenario(
        proposed_kill="MU breaks below $245",
        proposed_catalyst="Q4 earnings June 24",
        current_price=268.50,
        direction=Direction.long,
    )
    assert stop == 245.0
    assert entry == 268.50
    # target = entry + 2.5 * (entry - stop) = 268.5 + 2.5 * 23.5 = 327.25
    assert target == 327.25


def test_parse_levels_short():
    """Extract price levels from short scenario."""
    entry, stop, target = parse_levels_from_scenario(
        proposed_kill="KRE breaks above $52",
        proposed_catalyst="CRE maturity wave Q3",
        current_price=48.20,
        direction=Direction.short,
    )
    assert stop == 52.0
    assert entry == 48.20
    # target = entry - 2.5 * (stop - entry) = 48.20 - 2.5 * 3.8 = 38.70
    assert target == 38.70


if __name__ == "__main__":
    test_long_basic_sizing()
    test_short_basic_sizing()
    test_r_multiple_floor_failure()
    test_cfd_breakeven()
    test_parse_levels_long()
    test_parse_levels_short()
    print("All Stage 4a tests passed.")
