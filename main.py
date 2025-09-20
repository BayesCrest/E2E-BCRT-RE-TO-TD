"""Command line interface for the momentum trading engine."""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from typing import Sequence

from momentum_trader.broker import PaperBroker
from momentum_trader.config import MomentumConfig, ensure_iterable
from momentum_trader.data import CSVPriceProvider, PriceDataProvider, YahooPriceProvider
from momentum_trader.engine import MomentumEngine
from momentum_trader.portfolio import Portfolio
from momentum_trader.strategy import MomentumStrategy

LOGGER = logging.getLogger(__name__)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - input validation
        raise argparse.ArgumentTypeError(
            "Dates must be supplied in ISO format (YYYY-MM-DD)."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Momentum trading automation")
    parser.add_argument(
        "--tickers",
        "-t",
        nargs="+",
        required=True,
        help="Universe of tickers to trade (space or comma separated).",
    )
    parser.add_argument("--start", type=parse_date, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--lookback", type=int, default=126, help="Lookback window in trading days for momentum calculation")
    parser.add_argument("--top-n", type=int, default=5, help="Number of symbols to hold")
    parser.add_argument("--rebalance-days", type=int, default=5, help="Rebalance frequency in calendar days")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital for the simulation")
    parser.add_argument("--min-momentum", type=float, default=0.0, help="Minimum momentum score required to open a position")
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.3,
        help="Maximum portfolio weight per symbol. Use together with --no-max-weight to disable.",
    )
    parser.add_argument("--no-max-weight", action="store_true", help="Disable per-symbol weight caps")
    parser.add_argument("--transaction-cost", type=float, default=0.0005, help="Transaction cost percentage applied on fills")
    parser.add_argument("--csv-dir", type=str, help="Optional directory containing CSV files for offline data")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser


def build_provider(args: argparse.Namespace) -> PriceDataProvider:
    if args.csv_dir:
        LOGGER.info("Using CSV data provider (%s)", args.csv_dir)
        return CSVPriceProvider(args.csv_dir)
    LOGGER.info("Using Yahoo Finance data provider")
    return YahooPriceProvider()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    provider = build_provider(args)

    tickers = ensure_iterable(args.tickers)
    max_weight = None if args.no_max_weight else args.max_weight

    config = MomentumConfig(
        tickers=tickers,
        lookback_window=args.lookback,
        top_n=args.top_n,
        rebalance_frequency=timedelta(days=args.rebalance_days),
        capital=args.capital,
        max_weight=max_weight,
        min_momentum=args.min_momentum,
        transaction_cost=args.transaction_cost,
    )
    config.validate()

    strategy = MomentumStrategy(
        lookback_window=config.lookback_window,
        top_n=config.top_n,
        min_momentum=config.min_momentum,
        max_weight=config.max_weight,
    )
    portfolio = Portfolio(config.capital)
    broker = PaperBroker(portfolio, config.transaction_cost)
    engine = MomentumEngine(config, provider, strategy, broker, portfolio)

    LOGGER.info(
        "Starting momentum backtest for %s between %s and %s",
        ", ".join(config.normalised_tickers()),
        args.start,
        args.end,
    )
    result = engine.run(args.start, args.end)

    equity_curve = result.portfolio_history
    equity_curve["returns"] = equity_curve["equity"].pct_change().fillna(0.0)
    equity_curve["cumulative"] = (1 + equity_curve["returns"]).cumprod()

    LOGGER.info("Backtest completed. Final equity: %.2f", equity_curve["equity"].iloc[-1])
    print("\nPerformance summary (tail):")
    print(equity_curve.tail())


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
