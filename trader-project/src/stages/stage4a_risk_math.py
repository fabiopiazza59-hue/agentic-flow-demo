"""Stage 4a — Deterministic Risk Math.

Pure Python. No LLM. Given a scenario with entry/stop/target and account config,
compute position sizing, R-multiples, portfolio heat, CFD financing, correlation.
Fully deterministic: same inputs → identical outputs.
"""

from __future__ import annotations

import math

from src.config.settings import AccountConfig
from src.models.schemas import (
    Direction,
    Horizon,
    OpenPosition,
    RiskMath,
)


# Horizon to approximate holding days for CFD breakeven calc
_HORIZON_DAYS = {
    Horizon.days: 5,
    Horizon.weeks: 21,
    Horizon.months: 63,
}


def compute_risk_math(
    entry: float,
    stop: float,
    target: float,
    direction: Direction,
    current_price: float,
    proposed_horizon: Horizon,
    account: AccountConfig,
    open_positions: list[OpenPosition] | None = None,
) -> RiskMath:
    """Compute all risk math fields for a single scenario.

    Uses entry (not current_price) for sizing — entry is the planned level.
    """
    if open_positions is None:
        open_positions = []

    is_long = direction in (Direction.long, Direction.pair)

    # Stop and target distances
    if is_long:
        stop_distance = entry - stop
        target_distance = target - entry
    else:
        stop_distance = stop - entry
        target_distance = entry - target

    stop_distance_pct = abs(stop_distance / entry) if entry else 0
    target_distance_pct = abs(target_distance / entry) if entry else 0

    # R-multiple
    r_multiple = (target_distance / stop_distance) if stop_distance > 0 else 0

    # Position sizing: risk max_loss_per_trade_pct of account
    max_loss_usd = account.total_capital_usd * account.max_loss_per_trade_pct
    if stop_distance_pct > 0:
        position_size_usd = max_loss_usd / stop_distance_pct
    else:
        position_size_usd = 0

    # Cap position at available capital (no over-leverage beyond 5x for safety)
    max_notional = account.total_capital_usd * 5
    position_size_usd = min(position_size_usd, max_notional)

    position_size_units = position_size_usd / entry if entry else 0

    leverage_required = position_size_usd / account.total_capital_usd if account.total_capital_usd else 1.0

    max_loss_pct_account = (position_size_usd * stop_distance_pct) / account.total_capital_usd if account.total_capital_usd else 0

    # Portfolio heat: sum of existing risk + this trade
    existing_heat = sum(
        abs(p.entry - p.stop) / p.entry * abs(p.entry * 1)  # simplified
        for p in open_positions
        if p.entry > 0
    )
    # Simplify: use max_loss_pct_account as this trade's heat contribution
    current_heat = sum(
        _position_heat(p, account) for p in open_positions
    )
    portfolio_heat_after = current_heat + max_loss_pct_account

    # CFD overnight financing
    cfd_daily_cost = (position_size_usd * account.cfd_annual_financing_rate) / 365
    if target_distance_pct > 0 and cfd_daily_cost > 0:
        target_profit_usd = position_size_usd * target_distance_pct
        cfd_breakeven_days = int(target_profit_usd / cfd_daily_cost)
    else:
        cfd_breakeven_days = 999

    # Risk floor checks
    risk_floor_failures: list[str] = []
    if r_multiple < account.r_multiple_floor:
        risk_floor_failures.append(f"r_multiple_below_{account.r_multiple_floor}")
    if max_loss_pct_account > account.max_loss_per_trade_pct:
        risk_floor_failures.append("max_loss_exceeds_per_trade_limit")
    if portfolio_heat_after > account.max_portfolio_heat_pct:
        risk_floor_failures.append("portfolio_heat_exceeds_limit")
    if leverage_required > 5.0:
        risk_floor_failures.append("leverage_too_high")

    # CFD financing warning
    holding_days = _HORIZON_DAYS.get(proposed_horizon, 21)
    if holding_days > cfd_breakeven_days * 0.6:
        risk_floor_failures.append("cfd_financing_eats_profit")

    passes_risk_floor = len(risk_floor_failures) == 0

    return RiskMath(
        proposed_entry=entry,
        proposed_stop=stop,
        proposed_target=target,
        stop_distance_pct=round(stop_distance_pct, 4),
        target_distance_pct=round(target_distance_pct, 4),
        r_multiple=round(r_multiple, 2),
        position_size_usd=round(position_size_usd, 2),
        position_size_units=round(position_size_units, 2),
        leverage_required=round(leverage_required, 2),
        max_loss_usd=round(max_loss_usd, 2),
        max_loss_pct_account=round(max_loss_pct_account, 4),
        correlation_with_open_positions=None,  # requires price history — v0.2
        portfolio_heat_after=round(portfolio_heat_after, 4),
        cfd_overnight_cost_per_day=round(cfd_daily_cost, 4),
        cfd_breakeven_days=cfd_breakeven_days,
        passes_risk_floor=passes_risk_floor,
        risk_floor_failures=risk_floor_failures,
    )


def _position_heat(position: OpenPosition, account: AccountConfig) -> float:
    """Compute heat contribution of an open position as % of account."""
    if position.entry <= 0 or account.total_capital_usd <= 0:
        return 0
    stop_distance_pct = abs(position.entry - position.stop) / position.entry
    # Approximate position size from entry (simplified — real version tracks actual size)
    return stop_distance_pct * account.max_loss_per_trade_pct / stop_distance_pct if stop_distance_pct > 0 else 0


def parse_levels_from_scenario(proposed_kill: str, proposed_catalyst: str, current_price: float, direction: Direction) -> tuple[float, float, float]:
    """Extract entry/stop/target from scenario text.

    Heuristic: entry = current_price, stop = extracted from proposed_kill price level,
    target = entry + 2.5 * stop_distance (spec default).

    Returns (entry, stop, target).
    """
    entry = current_price
    is_long = direction in (Direction.long, Direction.pair)

    # Try to extract a price from proposed_kill (e.g., "MU breaks below $245")
    stop = _extract_price(proposed_kill)

    # Validate extracted stop makes sense for the direction and is in the
    # same order of magnitude as entry (catches commodity-price references
    # like "$70/lb uranium" when the stock is $55).
    if stop is not None:
        ratio = stop / entry if entry else 0
        if is_long and (stop >= entry or ratio < 0.5 or ratio > 1.0):
            stop = None  # nonsensical for a long — fall back
        elif not is_long and (stop <= entry or ratio > 1.5 or ratio < 1.0):
            stop = None  # nonsensical for a short — fall back

    if stop is None:
        # Fallback: 5% stop
        if is_long:
            stop = entry * 0.95
        else:
            stop = entry * 1.05

    stop_distance = abs(entry - stop)
    if is_long:
        target = entry + 2.5 * stop_distance
    else:
        target = entry - 2.5 * stop_distance

    return round(entry, 2), round(stop, 2), round(target, 2)


def _extract_price(text: str) -> float | None:
    """Extract a dollar price from text like 'below $245' or 'above $52'."""
    import re
    match = re.search(r'\$([0-9]+(?:\.[0-9]+)?)', text)
    if match:
        return float(match.group(1))
    return None
