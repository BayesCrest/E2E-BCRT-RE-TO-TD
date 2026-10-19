"""Implementation of the momentum ranking strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MomentumSignal:
    """Representation of the desired portfolio weights."""

    weights: Dict[str, float]
    scores: Dict[str, float]


class MomentumStrategy:
    """Rank securities by momentum and assign target weights."""

    def __init__(
        self,
        lookback_window: int,
        top_n: int | None = None,
        min_momentum: float = 0.0,
        max_weight: float | None = None,
    ) -> None:
        if lookback_window <= 0:
            raise ValueError("lookback_window must be positive")
        self.lookback_window = lookback_window
        self.top_n = top_n
        self.min_momentum = min_momentum
        self.max_weight = max_weight

    def evaluate(self, history: pd.DataFrame) -> MomentumSignal:
        """Compute a momentum signal using historical prices."""

        if history.empty:
            raise ValueError("Price history is empty.")
        if len(history) < self.lookback_window:
            raise ValueError(
                "Insufficient history to evaluate momentum. "
                f"Required {self.lookback_window} observations, got {len(history)}."
            )

        # Calculate cumulative return over the lookback horizon.
        window = history.tail(self.lookback_window)
        # Use log returns to stabilise distributions.
        log_returns = np.log(window / window.shift(1))
        momentum_scores = log_returns.sum(skipna=True)
        momentum_scores = momentum_scores.replace({np.inf: np.nan, -np.inf: np.nan}).dropna()

        returns = window.pct_change().dropna(how="all")
        volatility = returns.std(skipna=True)

        # Filter out securities below the minimum score threshold.
        filtered = momentum_scores[momentum_scores >= self.min_momentum]
        if filtered.empty:
            return MomentumSignal(weights={}, scores=momentum_scores.to_dict())

        ranked = filtered.sort_values(ascending=False)
        if self.top_n is not None:
            ranked = ranked.head(self.top_n)

        weights = self._compute_weights(ranked, volatility)
        return MomentumSignal(weights=weights, scores=momentum_scores.to_dict())

    def _compute_weights(
        self, ranked_scores: pd.Series, volatility: pd.Series | None
    ) -> Dict[str, float]:
        if ranked_scores.empty:
            return {}

        inverse_vol = self._inverse_volatility_weights(ranked_scores.index, volatility)
        if inverse_vol is None:
            equal_weight = 1.0 / len(ranked_scores)
            weights = {ticker: equal_weight for ticker in ranked_scores.index}
        else:
            # Normalise the inverse volatility weights so that they sum to one.
            inv_vol_sum = sum(inverse_vol.values())
            weights = {ticker: weight / inv_vol_sum for ticker, weight in inverse_vol.items()}

        if self.max_weight is not None:
            weights = self._apply_max_weight(weights, self.max_weight)

        return weights

    def _inverse_volatility_weights(
        self, tickers: Iterable[str], volatility: pd.Series | None
    ) -> Dict[str, float] | None:
        if volatility is None or volatility.empty:
            return None

        volatilities: Dict[str, float] = {}
        for ticker in tickers:
            vol = float(volatility.get(ticker, float("nan")))
            if np.isnan(vol) or vol <= 0:
                return None
            volatilities[ticker] = vol

        return {ticker: 1.0 / vol for ticker, vol in volatilities.items()}

    @staticmethod
    def _apply_max_weight(weights: Dict[str, float], max_weight: float) -> Dict[str, float]:
        """Cap long-only target weights, leaving the unallocated balance in cash.

        Inputs from _compute_weights sum to one. Do not renormalise after
        clipping: doing so can violate the cap, including when there are too
        few eligible securities to invest all capital. Deliberately avoid
        redistribution so this operation never increases an input weight.
        """
        return {ticker: min(weight, max_weight) for ticker, weight in weights.items()}
