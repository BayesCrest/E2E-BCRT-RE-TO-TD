"""Configuration utilities for the E2E research-to-trading pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataSourceConfig:
    """Parameters used to download market and alternative datasets."""

    symbols: List[str]
    start_date: str
    end_date: str
    interval: str = "1d"
    market_data_dir: Path = Path("data/raw/market")
    alternative_data_dir: Path = Path("data/raw/alternative")
    cache: bool = True


@dataclass
class FeatureConfig:
    """Configuration describing feature engineering behavior."""

    technical_indicators: Optional[List[str]] = None
    sentiment_model: str = "gpt-4"
    macro_variables: Optional[List[str]] = None
    lookback_window: int = 60


@dataclass
class AgentConfig:
    """Configuration for training reinforcement-learning agents."""

    algorithm: str = "PPO"
    total_timesteps: int = 250_000
    env_name: str = "TradingEnv-v0"
    policy_kwargs: dict = field(default_factory=dict)
    checkpoint_dir: Path = Path("artifacts/agents")


@dataclass
class BacktestConfig:
    """Configuration for running backtests."""

    initial_cash: float = 100_000.0
    slippage_bps: float = 0.5
    commission_bps: float = 1.0
    evaluation_dir: Path = Path("artifacts/evaluation")


@dataclass
class ProjectConfig:
    """Top level project configuration container."""

    data: DataSourceConfig
    features: FeatureConfig
    agent: AgentConfig
    backtest: BacktestConfig
    base_dir: Path = Path(".")

    def ensure_directories(self) -> None:
        """Create directories for all configured paths if they don't exist."""

        paths = [
            self.data.market_data_dir,
            self.data.alternative_data_dir,
            self.agent.checkpoint_dir,
            self.backtest.evaluation_dir,
        ]
        for path in paths:
            full_path = self.base_dir / path
            full_path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = ProjectConfig(
    data=DataSourceConfig(symbols=["SPY"], start_date="2015-01-01", end_date="2023-12-31"),
    features=FeatureConfig(technical_indicators=["sma", "ema", "rsi"], macro_variables=["CPI", "GDP"]),
    agent=AgentConfig(),
    backtest=BacktestConfig(),
)
