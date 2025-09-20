"""Data ingestion layer for market and alternative datasets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

LOGGER = logging.getLogger(__name__)


@dataclass
class DataIngestionResult:
    """Container describing the outcome of a data ingestion job."""

    market_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    alternative_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Serialize metadata to JSON for lineage tracking."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.metadata, fp, indent=2, default=str)


class MarketDataIngestor:
    """Utility responsible for downloading or synthesizing market datasets."""

    def __init__(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        data_dir: Path = Path("data/raw/market"),
        use_cache: bool = True,
    ) -> None:
        self.symbols = list(symbols)
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # region public API
    def run(self) -> Dict[str, pd.DataFrame]:
        """Fetch price history for all configured symbols."""

        results: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            results[symbol] = self._fetch_symbol(symbol)
            cache_path = self.data_dir / f"{symbol}_{self.interval}.parquet"
            results[symbol].to_parquet(cache_path)
            LOGGER.info("Stored market data for %s at %s", symbol, cache_path)
        return results

    # endregion

    # region internal helpers
    def _fetch_symbol(self, symbol: str) -> pd.DataFrame:
        """Fetch a single symbol either via yfinance or by synthesizing data."""

        cache_path = self.data_dir / f"{symbol}_{self.interval}.parquet"
        if self.use_cache and cache_path.exists():
            LOGGER.debug("Loading cached market data for %s", symbol)
            return pd.read_parquet(cache_path)

        if yf is not None:
            LOGGER.debug("Downloading %s from yfinance", symbol)
            data = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,
            )
            data.index = pd.DatetimeIndex(data.index).tz_localize(None)
        else:  # Offline fallback
            LOGGER.warning(
                "yfinance not available; generating synthetic price series for %s", symbol
            )
            dates = pd.date_range(self.start_date, self.end_date, freq=self.interval.upper())
            base_price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
            data = pd.DataFrame(
                {
                    "Open": base_price * (1 + np.random.normal(0, 0.01, len(dates))),
                    "High": base_price * (1 + np.random.normal(0.01, 0.01, len(dates))),
                    "Low": base_price * (1 + np.random.normal(-0.01, 0.01, len(dates))),
                    "Close": base_price,
                    "Adj Close": base_price,
                    "Volume": np.random.randint(1_000_000, 5_000_000, len(dates)),
                },
                index=dates,
            )
        data = self.clean_price_data(data)
        data["symbol"] = symbol
        return data

    # endregion

    @staticmethod
    def clean_price_data(data: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning steps applied to price data."""

        cleaned = data.copy()
        cleaned = cleaned.sort_index()
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        numeric_cols = cleaned.select_dtypes(include=["float", "int"]).columns
        cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
        cleaned[numeric_cols] = cleaned[numeric_cols].fillna(method="ffill").fillna(method="bfill")
        return cleaned


class AlternativeDataIngestor:
    """Fetch alternative datasets such as news sentiment and on-chain metrics."""

    def __init__(
        self,
        providers: Optional[Dict[str, Dict[str, str]]] = None,
        data_dir: Path = Path("data/raw/alternative"),
        use_cache: bool = True,
    ) -> None:
        self.providers = providers or {
            "news_sentiment": {"source": "synthetic", "frequency": "1d"},
            "onchain_flows": {"source": "synthetic", "frequency": "1d"},
        }
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def run(self, dates: Iterable[pd.Timestamp]) -> Dict[str, pd.DataFrame]:
        """Generate or download alternative datasets aligned with `dates`."""

        aligned_index = pd.DatetimeIndex(list(dates))
        results: Dict[str, pd.DataFrame] = {}
        for name, meta in self.providers.items():
            cache_path = self.data_dir / f"{name}.parquet"
            if self.use_cache and cache_path.exists():
                LOGGER.debug("Loading cached alternative data for %s", name)
                results[name] = pd.read_parquet(cache_path)
                continue

            LOGGER.info("Generating alternative dataset '%s'", name)
            data = pd.DataFrame(index=aligned_index)
            rng = np.random.default_rng(seed=hash(name) % 2**32)
            data[f"{name}_score"] = rng.normal(loc=0.0, scale=1.0, size=len(aligned_index))
            data[f"{name}_zscore"] = (data[f"{name}_score"] - data[f"{name}_score"].mean()) / data[
                f"{name}_score"
            ].std()
            data["source"] = meta.get("source", "unknown")
            data.to_parquet(cache_path)
            results[name] = data
        return results


def run_data_ingestion(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
    use_cache: bool = True,
) -> DataIngestionResult:
    """High-level convenience wrapper to execute full ingestion."""

    market_ingestor = MarketDataIngestor(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        use_cache=use_cache,
    )
    market_data = market_ingestor.run()

    combined_index = pd.DatetimeIndex(sorted({idx for df in market_data.values() for idx in df.index}))
    alternative_ingestor = AlternativeDataIngestor(use_cache=use_cache)
    alternative_data = alternative_ingestor.run(combined_index)

    metadata = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
        "generated_at": datetime.utcnow().isoformat(),
        "market_files": {
            symbol: str(market_ingestor.data_dir / f"{symbol}_{interval}.parquet") for symbol in symbols
        },
        "alternative_files": {
            name: str(alternative_ingestor.data_dir / f"{name}.parquet") for name in alternative_data.keys()
        },
    }

    result = DataIngestionResult(
        market_data=market_data,
        alternative_data=alternative_data,
        metadata=metadata,
    )
    result.to_json(Path("data/raw/metadata.json"))
    return result


__all__ = [
    "AlternativeDataIngestor",
    "DataIngestionResult",
    "MarketDataIngestor",
    "run_data_ingestion",
]
