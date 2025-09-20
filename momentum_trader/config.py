"""Configuration objects for the momentum trading system.

This module hosts the dataclasses that describe runtime configuration for the
momentum trading engine.  The configuration is intentionally explicit so the
behaviour of the system can be reproduced or audited later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, List, Sequence


@dataclass(slots=True)
class MomentumConfig:
    """High level configuration for the trading engine.

    Attributes
    ----------
    tickers:
        Iterable universe of symbols to trade.  The trading engine will fetch
        data for all symbols and apply the momentum ranking across the full
        universe.
    lookback_window:
        Number of trading days used for calculating momentum scores.  Larger
        windows provide smoother momentum estimates but react slower to changes.
    top_n:
        How many symbols should be selected for long positions.  If the number
        is larger than the available tickers the full universe is traded.
    rebalance_frequency:
        How often the portfolio should be rebalanced.  The value is expressed
        as a :class:`datetime.timedelta` for clarity and works with most
        backtesting regimes.
    capital:
        Initial cash level used for the simulation.
    max_weight:
        Optional upper bound on the weight of a single symbol.  This prevents
        the strategy from becoming overly concentrated in a single instrument.
    min_momentum:
        Minimum momentum score a symbol must exhibit to become a candidate for
        trading.  Scores below this threshold result in a zero weight.
    transaction_cost:
        Flat percentage of the traded notional that will be charged as
        transaction cost.  For real deployments this can be replaced by a more
        elaborate execution model.
    """

    tickers: Sequence[str]
    lookback_window: int = 126
    top_n: int | None = 5
    rebalance_frequency: timedelta = field(
        default_factory=lambda: timedelta(days=5)
    )
    capital: float = 100_000.0
    max_weight: float | None = 0.3
    min_momentum: float = 0.0
    transaction_cost: float = 0.0005

    def validate(self) -> None:
        """Validate configuration parameters.

        This function raises :class:`ValueError` when an invalid configuration
        is detected which helps catching issues early.
        """

        if not self.tickers:
            raise ValueError("No tickers were supplied. The universe cannot be empty.")
        if self.lookback_window <= 0:
            raise ValueError("Lookback window must be positive.")
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError("top_n must be positive when provided.")
        if self.rebalance_frequency <= timedelta(0):
            raise ValueError("rebalance_frequency must be a positive duration.")
        if self.capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if self.max_weight is not None and not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be within (0, 1].")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost cannot be negative.")

    def normalised_tickers(self) -> List[str]:
        """Return the tickers as uppercase strings."""

        return [ticker.upper() for ticker in self.tickers]


@dataclass(slots=True)
class DateRange:
    """Helper describing the backtest period."""

    start: date
    end: date

    def validate(self) -> None:
        if self.end <= self.start:
            raise ValueError("The end date must be after the start date.")


def ensure_iterable(value: Iterable[str] | str) -> List[str]:
    """Coerce tickers into a list.

    The CLI accepts both comma separated strings and repeated flags.  This
    helper simplifies the conversion into a canonical in-memory format.
    """

    if isinstance(value, str):
        return [part.strip().upper() for part in value.split(",") if part.strip()]
    return [str(item).upper() for item in value]
