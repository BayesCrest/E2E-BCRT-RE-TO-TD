"""Backtesting utilities for evaluating trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]


def simple_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 0.5,
    commission_bps: float = 1.0,
) -> BacktestResult:
    """Execute a basic long-only backtest using allocation signals between -1 and 1."""

    prices = prices.ffill().dropna()
    returns = prices.pct_change().fillna(0.0)
    allocation = signals.clip(-1.0, 1.0).reindex(prices.index).fillna(method="ffill").fillna(0.0)
    turnover = allocation.diff().abs().fillna(0.0)

    gross_returns = allocation.shift().fillna(0.0) * returns
    costs = (slippage_bps + commission_bps) / 10_000.0 * turnover
    net_returns = gross_returns - costs

    equity_curve = (1 + net_returns).cumprod() * initial_cash
    trades = pd.DataFrame({
        "price": prices,
        "allocation": allocation,
        "turnover": turnover,
        "net_returns": net_returns,
        "equity": equity_curve,
    })

    metrics = {
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        "volatility": float(net_returns.std() * np.sqrt(252)),
        "sharpe": float(net_returns.mean() / (net_returns.std() + 1e-6) * np.sqrt(252)),
        "max_drawdown": float(_max_drawdown(equity_curve)),
    }
    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics)


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


__all__ = ["BacktestResult", "simple_backtest"]
