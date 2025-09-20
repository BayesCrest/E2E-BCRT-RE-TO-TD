"""Technical indicator feature engineering utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

try:  # pragma: no cover
    import ta
except ImportError:  # pragma: no cover
    ta = None


def _ensure_ta_available() -> None:
    if ta is None:
        raise ImportError(
            "The 'ta' package is required for advanced technical indicators. Install it via 'pip install ta'."
        )


def moving_average_features(data: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Compute moving averages for specified windows."""

    features = data[["Close"]].copy()
    for window in windows:
        features[f"sma_{window}"] = data["Close"].rolling(window=window, min_periods=1).mean()
        features[f"ema_{window}"] = data["Close"].ewm(span=window, min_periods=1, adjust=False).mean()
    return features.drop(columns=["Close"]).dropna(how="all")


def volatility_features(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate rolling volatility and returns-based features."""

    returns = data["Close"].pct_change().fillna(0.0)
    rolling_vol = returns.rolling(window).std().fillna(method="bfill")
    features = pd.DataFrame(
        {
            "returns": returns,
            f"rolling_vol_{window}": rolling_vol,
            f"rolling_sharpe_{window}": returns.rolling(window).mean() / (rolling_vol + 1e-6),
        },
        index=data.index,
    )
    return features


def momentum_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, and other momentum-driven indicators."""

    _ensure_ta_available()
    indicator = ta.momentum.RSIIndicator(close=data["Close"], window=14)
    macd = ta.trend.MACD(close=data["Close"])
    return pd.DataFrame(
        {
            "rsi": indicator.rsi(),
            "macd": macd.macd(),
            "macd_signal": macd.macd_signal(),
            "macd_diff": macd.macd_diff(),
        },
        index=data.index,
    )


def assemble_technical_features(
    data: pd.DataFrame,
    indicator_set: Optional[List[str]] = None,
    windows: Iterable[int] = (5, 20, 60),
) -> pd.DataFrame:
    """Build a combined technical feature matrix given raw OHLCV data."""

    indicator_set = indicator_set or ["sma", "ema", "volatility", "momentum"]
    feature_frames: List[pd.DataFrame] = []

    if any(ind in indicator_set for ind in ("sma", "ema")):
        feature_frames.append(moving_average_features(data, windows))
    if "volatility" in indicator_set:
        feature_frames.append(volatility_features(data))
    if "momentum" in indicator_set:
        try:
            feature_frames.append(momentum_features(data))
        except ImportError:
            # Provide graceful degradation when ta is unavailable.
            feature_frames.append(
                pd.DataFrame(
                    {
                        "rsi": data["Close"].pct_change().rolling(14).mean(),
                        "macd": data["Close"].diff(12) - data["Close"].diff(26),
                    },
                    index=data.index,
                )
            )

    combined = pd.concat(feature_frames, axis=1).dropna(how="all")
    combined.columns = pd.Index([col.lower() for col in combined.columns])
    return combined


__all__ = [
    "assemble_technical_features",
    "momentum_features",
    "moving_average_features",
    "volatility_features",
]
