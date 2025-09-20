"""Streamlit dashboard for monitoring the research-to-trading pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ..evaluation.metrics import compute_performance_metrics


def load_equity_curve(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    return df.set_index("date")["equity"] if "equity" in df.columns else df.squeeze()


def main() -> None:
    st.set_page_config(page_title="E2E Research to Trading", layout="wide")
    st.title("E2E Research to Trading Dashboard")

    st.sidebar.header("Data Sources")
    market_metadata_path = Path("data/raw/metadata.json")
    if market_metadata_path.exists():
        metadata = json.loads(market_metadata_path.read_text())
        st.sidebar.json(metadata)
    else:
        st.sidebar.info("Run data ingestion to populate metadata.")

    st.header("Performance Overview")
    equity_path = Path("artifacts/evaluation/equity.parquet")
    equity = load_equity_curve(equity_path)
    if equity.empty:
        st.warning("No evaluation artifacts found. Execute an evaluation workflow first.")
    else:
        st.line_chart(equity)
        metrics = compute_performance_metrics(equity)
        st.subheader("Key Metrics")
        st.json(metrics)

    st.header("Latest Research Notes")
    report_path = Path("artifacts/evaluation/research_report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.write(f"### {report.get('title', 'Research Update')}")
        st.write(report.get("summary", ""))
        st.info(f"Recommended Action: {report.get('recommended_action', 'N/A')}")
        for point in report.get("supporting_points", []):
            st.markdown(f"- {point}")
    else:
        st.info("Generate a research report to see it here.")


if __name__ == "__main__":  # pragma: no cover
    main()
