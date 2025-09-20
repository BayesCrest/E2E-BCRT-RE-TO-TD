"""Trading engine orchestrating the data, strategy and execution layers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from .broker import PaperBroker
from .config import MomentumConfig
from .data import PriceDataProvider
from .portfolio import Portfolio
from .strategy import MomentumStrategy

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EngineResult:
    history: pd.DataFrame
    portfolio_history: pd.DataFrame


class MomentumEngine:
    """Coordinate data retrieval, signal generation and execution."""

    def __init__(
        self,
        config: MomentumConfig,
        provider: PriceDataProvider,
        strategy: MomentumStrategy,
        broker: PaperBroker,
        portfolio: Portfolio,
    ) -> None:
        self.config = config
        self.provider = provider
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio

    def run(self, start: date, end: date) -> EngineResult:
        LOGGER.info("Fetching price history for %s", ", ".join(self.config.tickers))
        history = self.provider.get_price_history(self.config.tickers, start, end)
        history = history.dropna(how="all")
        if history.empty:
            raise ValueError("Received empty price history.")

        portfolio_snapshots: list[dict[str, float]] = []
        self._available_dates = history.index
        schedule = self._build_schedule(history.index[0].date(), history.index[-1].date())
        LOGGER.info("Running backtest between %s and %s (%d rebalances)", start, end, len(schedule))

        # Record the initial portfolio state before any trades occur.
        warmup_index = self.config.lookback_window - 1
        initial_prices = history.iloc[warmup_index]
        self.portfolio.update_market_prices(initial_prices.to_dict())
        portfolio_snapshots.append(
            {
                "date": history.index[warmup_index],
                "cash": self.portfolio.cash,
                "equity": self.portfolio.equity,
            }
        )

        for timestamp in schedule:
            prices = history.loc[:timestamp].iloc[-1]
            self.portfolio.update_market_prices(prices.to_dict())
            LOGGER.info("Evaluating strategy on %s", timestamp.date())
            signal = self.strategy.evaluate(history.loc[:timestamp])
            orders = self.portfolio.desired_orders(signal.weights, self.config.transaction_cost)
            report = self.broker.execute(orders)
            portfolio_snapshots.append(
                {
                    "date": timestamp,
                    "cash": report.cash,
                    "equity": report.equity,
                }
            )

        portfolio_history = (
            pd.DataFrame(portfolio_snapshots).set_index("date").sort_index()
        )
        return EngineResult(history=history, portfolio_history=portfolio_history)

    def _build_schedule(self, start: date, end: date) -> list[pd.Timestamp]:
        available = sorted(
            dt for dt in self._available_dates if start <= dt.date() <= end
        )
        if not available:
            raise ValueError("No trading days available within the requested range.")

        min_index = self.config.lookback_window - 1
        if len(available) <= min_index:
            raise ValueError(
                "Not enough price history to warm up the strategy. "
                f"Need at least {self.config.lookback_window} observations."
            )

        eligible = available[min_index:]
        schedule = [eligible[0]]
        frequency = pd.Timedelta(self.config.rebalance_frequency)
        for current in eligible[1:]:
            if current - schedule[-1] >= frequency:
                schedule.append(current)

        if schedule[-1] != eligible[-1]:
            schedule.append(eligible[-1])
        return schedule

    @property
    def _available_dates(self) -> set[pd.Timestamp]:
        # This property is memoised by computing once and caching on the instance.
        if not hasattr(self, "__available_dates"):
            raise AttributeError("Price history not loaded yet. Run `run` first.")
        return getattr(self, "__available_dates")

    @_available_dates.setter
    def _available_dates(self, value: Iterable[pd.Timestamp]) -> None:
        setattr(self, "__available_dates", set(pd.to_datetime(list(value))))

    def prepare(self, start: date, end: date) -> pd.DataFrame:
        """Fetch data and expose to users that require custom processing."""

        history = self.provider.get_price_history(self.config.tickers, start, end)
        self._available_dates = history.index
        return history
