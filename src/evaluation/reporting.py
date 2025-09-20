"""Reporting utilities for summarizing evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import plotly.graph_objects as go

from .metrics import compute_performance_metrics


def generate_html_report(equity_curve, trades, output_path: Path) -> Dict[str, float]:
    """Create an interactive HTML report summarizing backtest results."""

    metrics = compute_performance_metrics(equity_curve)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity"))
    figure.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Portfolio Value")
    figure.write_html(str(output_path), include_plotlyjs="cdn")

    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(str(metrics))
    return metrics


__all__ = ["generate_html_report"]
