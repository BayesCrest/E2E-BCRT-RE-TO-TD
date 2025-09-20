"""Macro relationship modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class MacroModelResult:
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float


def build_macro_regression(
    features: pd.DataFrame,
    target: pd.Series,
    macro_variables: Optional[pd.DataFrame] = None,
) -> MacroModelResult:
    """Fit a linear model explaining the target returns via macro variables."""

    design_matrix = features.copy()
    if macro_variables is not None:
        design_matrix = design_matrix.join(macro_variables, how="left").fillna(method="ffill")

    design_matrix = design_matrix.dropna()
    target_aligned = target.loc[design_matrix.index]

    model = LinearRegression()
    model.fit(design_matrix, target_aligned)
    predictions = model.predict(design_matrix)
    residuals = target_aligned - predictions
    total_variance = ((target_aligned - target_aligned.mean()) ** 2).sum()
    explained_variance = total_variance - (residuals**2).sum()
    r_squared = explained_variance / total_variance if total_variance else 0.0

    return MacroModelResult(
        coefficients={name: float(coef) for name, coef in zip(design_matrix.columns, model.coef_)},
        intercept=float(model.intercept_),
        r_squared=float(r_squared),
    )


__all__ = ["MacroModelResult", "build_macro_regression"]
