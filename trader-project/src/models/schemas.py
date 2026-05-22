"""Pydantic models matching the YAML I/O contracts from the specs.

Each stage's input/output is a strict schema. Stages pass structured records,
never prose, never conversation.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──

class DirectionImplication(str, Enum):
    long_bias = "long_bias"
    short_bias = "short_bias"
    mixed = "mixed"
    neutral = "neutral"


class Horizon(str, Enum):
    days = "days"
    weeks = "weeks"
    months = "months"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Direction(str, Enum):
    long = "long"
    short = "short"
    pair = "pair"
    avoid = "avoid"


class InstrumentClass(str, Enum):
    equity = "equity"
    etf = "etf"
    commodity_etf = "commodity_etf"


class Verdict(str, Enum):
    LIVE = "LIVE"
    WATCH = "WATCH"
    KILL = "KILL"


class Crowding(str, Enum):
    long_crowded = "long_crowded"
    short_crowded = "short_crowded"
    under_owned = "under_owned"
    unknown = "unknown"


class EdgeSource(str, Enum):
    information = "information"
    interpretation = "interpretation"
    time_horizon = "time_horizon"
    behavioral = "behavioral"
    structural = "structural"
    none = "none"


class BearVerdict(str, Enum):
    dangerous = "dangerous"
    acceptable_risk = "acceptable_risk"
    strong_objection = "strong_objection"


class Decision(str, Enum):
    take = "take"
    pass_ = "pass"
    reduce_size = "reduce_size"
    defer = "defer"


class NetDirection(str, Enum):
    long_skew = "long_skew"
    short_skew = "short_skew"
    balanced = "balanced"


class PositionAction(str, Enum):
    HOLD = "HOLD"
    TRIM = "TRIM"
    EXIT = "EXIT"
    ADD = "ADD"


class TradeAction(str, Enum):
    BUY = "BUY"
    SELL_SHORT = "SELL_SHORT"


# ── Stage 1: Market Scan ──

class Evidence(BaseModel):
    claim: str
    source: str
    url: str
    date: date


class Theme(BaseModel):
    theme_id: str
    lens: str  # "journalist" or "macro_analyst"
    title: str = Field(max_length=80)
    summary: str
    direction_implication: DirectionImplication
    horizon: Horizon
    evidence: list[Evidence]
    contradicting_evidence: str
    confidence: Confidence


class LensMetadata(BaseModel):
    articles_consulted: int = 0
    sources_diversity: int = 0
    data_points_consulted: int = 0
    indicators_referenced: list[str] = Field(default_factory=list)


class Stage1Output(BaseModel):
    run_id: str
    run_date: date
    themes: list[Theme]
    lens_metadata: dict[str, LensMetadata] = Field(default_factory=dict)
    desk_note: str


# ── Stage 2: Ticker Discovery ──

class DataPoint(BaseModel):
    claim: str
    source: str
    url: str


class Scenario(BaseModel):
    scenario_id: str
    parent_theme_id: str
    direction: Direction
    instrument: str
    instrument_class: InstrumentClass
    thesis_summary: str
    proposed_horizon: Horizon
    proposed_catalyst: str
    proposed_kill: str
    key_data_points: list[DataPoint]
    why_this_expression: str
    alternative_expressions: list[str] = Field(default_factory=list)


class Stage2Output(BaseModel):
    run_id: str
    scenarios: list[Scenario]


# ── Stage 3: Triage Gate ──

class VerdictAnswers(BaseModel):
    trade_in_one_sentence: str
    variant_perception: str
    asymmetry: str
    catalyst_and_date: str
    crowding: Crowding
    edge_source: EdgeSource
    kill_scenario: str
    conviction: Confidence
    not_shorting_because: str = ""


class ScenarioVerdict(BaseModel):
    scenario_id: str
    verdict: Verdict
    one_line_reason: str = Field(max_length=120)
    answers: VerdictAnswers


class Stage3BatchSummary(BaseModel):
    batch_id: str
    run_timestamp: datetime
    scenarios_in: int
    verdicts: dict[str, int]  # {"LIVE": n, "WATCH": n, "KILL": n}
    top_3_by_conviction: list[str]
    desk_note: str


class Stage3Output(BaseModel):
    verdicts: list[ScenarioVerdict]
    batch_summary: Stage3BatchSummary


# ── Stage 4a: Risk Math ──

class RiskMath(BaseModel):
    proposed_entry: float
    proposed_stop: float
    proposed_target: float
    stop_distance_pct: float
    target_distance_pct: float
    r_multiple: float
    position_size_usd: float
    position_size_units: float
    leverage_required: float = 1.0
    max_loss_usd: float
    max_loss_pct_account: float
    correlation_with_open_positions: Optional[dict[str, float]] = None
    portfolio_heat_after: float
    cfd_overnight_cost_per_day: float
    cfd_breakeven_days: int
    passes_risk_floor: bool
    risk_floor_failures: list[str] = Field(default_factory=list)


# ── Stage 4b: Bear Challenger ──

class SpecificRisk(BaseModel):
    risk: str
    evidence: str
    probability_assessment: Confidence


class BearCase(BaseModel):
    thesis_against: str
    specific_risks: list[SpecificRisk]
    historical_precedent: str
    what_market_knows_that_we_dont: str
    recommended_size_haircut: float = Field(ge=0.0, le=1.0)
    bear_verdict: BearVerdict


# ── Stage 4 Combined ──

class EnrichedScenario(BaseModel):
    scenario_id: str
    stage3_verdict: Verdict
    current_price: float
    risk_math: RiskMath
    bear_case: Optional[BearCase] = None


class Stage4Output(BaseModel):
    run_id: str
    enriched_scenarios: list[EnrichedScenario]


# ── Stage 5: Synthesis Decision ──

class OpenPosition(BaseModel):
    ticker: str
    direction: Direction
    entry: float
    current_price: float
    stop: float
    unrealized_pnl_usd: float
    days_held: int


class AccountState(BaseModel):
    total_capital_usd: float
    available_capital_usd: float
    current_portfolio_heat_pct: float
    open_positions: list[OpenPosition] = Field(default_factory=list)


class TradeDecision(BaseModel):
    scenario_id: str
    decision: Decision
    final_direction: Optional[Direction] = None
    final_size_usd: float = 0.0
    rationale: str
    risks_acknowledged: str = ""
    defer_until: Optional[date] = None
    interaction_with_book: str = ""


class PositionCheck(BaseModel):
    ticker: str
    action: PositionAction
    action_size_pct: Optional[float] = None
    reason: str


class PortfolioAfter(BaseModel):
    projected_heat_pct: float
    new_positions_count: int
    net_direction: NetDirection


class Stage5Output(BaseModel):
    run_id: str
    decisions: list[TradeDecision]
    position_checks: list[PositionCheck] = Field(default_factory=list)
    portfolio_after: PortfolioAfter
    ic_note: str


# ── Stage 6: Action Table ──

class NewTradeRow(BaseModel):
    ticker: str
    action: TradeAction
    entry_price: float
    stop_loss: float
    target_price: float
    size_usd: float
    r_multiple: float
    reason: str = Field(max_length=150)
    scenario_id: str


class PositionManagementRow(BaseModel):
    ticker: str
    held_since: date
    current_pnl_usd: float
    action: PositionAction
    action_size_pct: Optional[float] = None
    reason: str = Field(max_length=100)


class Stage6Output(BaseModel):
    run_id: str
    generated_at: datetime
    pipeline_version: str = "0.2"
    new_trades: list[NewTradeRow]
    position_management: list[PositionManagementRow] = Field(default_factory=list)
    portfolio_after: PortfolioAfter
    ic_note: str
