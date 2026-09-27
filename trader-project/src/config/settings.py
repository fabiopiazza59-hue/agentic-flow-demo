"""User-configurable settings for the Seamaster pipeline."""

from pydantic import BaseModel


class AccountConfig(BaseModel):
    total_capital_usd: float = 2000.0
    max_loss_per_trade_pct: float = 0.02  # 2% of account
    max_portfolio_heat_pct: float = 0.06  # 6% total risk
    r_multiple_floor: float = 2.0
    max_correlation_same_direction: float = 0.7
    cfd_annual_financing_rate: float = 0.07  # 7% typical CFD overnight rate
    drawdown_size_multiplier: float = 0.7  # reduce size when in >5% drawdown


class PipelineConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    stage3_temperature: float = 0.2
    stage4b_temperature: float = 0.4
    stage5_temperature: float = 0.3
    journalist_window_days: int = 7
    macro_window_days: int = 30
    max_themes_per_lens: int = 5
    max_scenarios_per_theme: int = 5


class Settings(BaseModel):
    account: AccountConfig = AccountConfig()
    pipeline: PipelineConfig = PipelineConfig()
