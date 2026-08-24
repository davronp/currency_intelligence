"""src/ml/metrics.py.

Pure forecast-evaluation metrics.

Each function takes array-like actuals/predictions and returns a plain
float, with no dependency on Prophet or any model object, so the
scoring logic can be unit-tested in isolation and fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


def _series(values: Iterable[float]) -> pd.Series:
    """Coerce *values* to a float Series with a positional index.

    Building from ``list(values)`` forces a fresh RangeIndex so that
    arithmetic between two inputs aligns by position, not by a caller's
    index labels.
    """
    return pd.Series(list(values), dtype="float64")


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean absolute error."""
    a = _series(actual)
    p = _series(predicted)
    return float((a - p).abs().mean())


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean absolute percentage error, in percent.

    Rows where the actual is zero are dropped to avoid division by zero.
    Returns NaN if no non-zero actuals remain.
    """
    a = _series(actual)
    p = _series(predicted)
    nonzero = a != 0
    if not nonzero.any():
        return float("nan")
    a = a[nonzero]
    p = p[nonzero]
    return float(((a - p).abs() / a.abs()).mean() * 100)


def coverage(
    actual: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
) -> float:
    """Percentage of actuals falling within the [lower, upper] interval.

    This measures how well-calibrated the forecast's uncertainty band is:
    for a 95% interval, a coverage near 95 means the band is honest.
    """
    a = _series(actual)
    lo = _series(lower)
    hi = _series(upper)
    if a.empty:
        return float("nan")
    within = (a >= lo) & (a <= hi)
    return float(within.mean() * 100)
