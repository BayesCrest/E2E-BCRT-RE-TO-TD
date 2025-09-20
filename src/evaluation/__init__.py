"""Evaluation package with backtesting and reporting utilities."""

from .backtest import BacktestResult, simple_backtest
from .metrics import compute_performance_metrics
from .reporting import generate_html_report

__all__ = [
    "BacktestResult",
    "compute_performance_metrics",
    "generate_html_report",
    "simple_backtest",
]
