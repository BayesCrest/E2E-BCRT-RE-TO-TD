"""Performance metrics utilities for trading strategies."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_performance_metrics(equity_curve: pd.Series, frequency: int = 252) -> Dict[str, float]:
    returns = equity_curve.pct_change().fillna(0.0)
    cumulative_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    volatility = float(returns.std() * np.sqrt(frequency))
    sharpe = float(returns.mean() / (returns.std() + 1e-6) * np.sqrt(frequency))
    downside = returns[returns < 0].std() * np.sqrt(frequency)
    sortino = float(returns.mean() / (downside + 1e-6) * np.sqrt(frequency))
    max_equity = equity_curve.cummax()
    drawdown = equity_curve / max_equity - 1
    max_drawdown = float(drawdown.min())
    calmar = float(returns.mean() * frequency / (abs(max_drawdown) + 1e-6))
    return {
        "cumulative_return": cumulative_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


__all__ = ["compute_performance_metrics"]
