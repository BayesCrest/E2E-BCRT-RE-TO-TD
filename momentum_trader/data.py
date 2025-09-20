"""Data loading facilities for the momentum trading system."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


class DataError(RuntimeError):
    """Raised when price data cannot be retrieved."""

class PriceDataProvider:
    """Interface for fetching historical price data."""

    def get_price_history(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        """Return a dataframe indexed by date with adjusted close prices."""

        raise NotImplementedError


class YahooPriceProvider(PriceDataProvider):
    """Download historical prices from Yahoo Finance using :mod:`yfinance`."""

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def get_price_history(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - informative error path
            raise DataError(
                "yfinance is required for YahooPriceProvider. Install it with `pip install yfinance`."
            ) from exc

        # yfinance expects strings, enforce uppercase to keep consistency.
        normalised = [ticker.upper() for ticker in tickers]
        data = yf.download(
            tickers=normalised,
            start=start,
            end=end,
            auto_adjust=self.auto_adjust,
            progress=False,
        )
        if data.empty:
            raise DataError("No data returned from Yahoo Finance. Check symbols and dates.")

        # `yf.download` returns a multi-index when multiple tickers are requested.
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Adj Close" if "Adj Close" in data.columns.levels[0] else "Close"]
        else:
            # Single series result – wrap in a frame for consistency
            data = data.to_frame(name=normalised[0])

        data.index = pd.to_datetime(data.index)
        data = data.sort_index().ffill().dropna(how="all")
        return data


class CSVPriceProvider(PriceDataProvider):
    """Load data from CSV files located in a directory."""

    def __init__(self, directory: str | Path, date_column: str = "Date") -> None:
        self.directory = Path(directory)
        self.date_column = date_column

    def get_price_history(
        self, tickers: Sequence[str], start: date, end: date
    ) -> pd.DataFrame:
        frames: list[pd.Series] = []
        for ticker in tickers:
            path = self.directory / f"{ticker.upper()}.csv"
            if not path.exists():
                raise DataError(f"CSV file for ticker {ticker} not found at {path!s}.")
            frame = pd.read_csv(path, parse_dates=[self.date_column])
            frame = frame.set_index(self.date_column)
            column = self._select_price_column(frame)
            series = column.loc[(column.index >= pd.Timestamp(start)) & (column.index <= pd.Timestamp(end))]
            series.name = ticker.upper()
            frames.append(series)

        if not frames:
            raise DataError("No CSV files were loaded.")

        data = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
        return data

    @staticmethod
    def _select_price_column(frame: pd.DataFrame) -> pd.Series:
        for candidate in ("Adj Close", "Close", "Price"):
            if candidate in frame.columns:
                return frame[candidate]
        raise DataError(
            "CSV file must contain an 'Adj Close', 'Close' or 'Price' column with price data."
        )


def align_dataframes(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Align and combine multiple dataframes on their indices."""

    data = pd.concat(frames, axis=1)
    data = data.sort_index().ffill().dropna(how="all")
    return data
