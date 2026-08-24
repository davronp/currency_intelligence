"""Tests for forecast metrics and backtesting.

The backtest tests inject a stub forecaster so they never fit Prophet:
that keeps them fast and lets them run in CI without a compiled Stan
backend.
"""

from __future__ import annotations

import math

import pandas as pd

from src.ml.forecast import backtest_pair
from src.ml.metrics import coverage, mae, mape


def _make_gold_df(pair: str, n: int, rate: float = 1.0) -> pd.DataFrame:
    """Build a constant-rate gold-layer frame for one pair."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "currency_pair": pair,
            "rate": [rate] * n,
        }
    )


def _perfect_forecaster(train_df: pd.DataFrame, holdout_ds: pd.Series) -> pd.DataFrame:
    """Predict the last training value exactly (perfect on a constant series)."""
    y = train_df["y"].iloc[-1]
    ds = pd.to_datetime(list(holdout_ds))
    return pd.DataFrame({"ds": ds, "yhat": y, "yhat_lower": y - 1, "yhat_upper": y + 1})


def _biased_forecaster(train_df: pd.DataFrame, holdout_ds: pd.Series) -> pd.DataFrame:
    """Predict last value + 10 with a tight band that excludes the actual."""
    pred = train_df["y"].iloc[-1] + 10
    ds = pd.to_datetime(list(holdout_ds))
    return pd.DataFrame({"ds": ds, "yhat": pred, "yhat_lower": pred - 1, "yhat_upper": pred + 1})


class TestMae:
    def test_zero_error(self) -> None:
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self) -> None:
        assert mae([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == 1.0

    def test_independent_of_input_index(self) -> None:
        actual = pd.Series([1.0, 2.0], index=[10, 20])
        predicted = pd.Series([2.0, 3.0], index=[99, 100])
        assert mae(actual, predicted) == 1.0


class TestMape:
    def test_zero_error(self) -> None:
        assert mape([100.0, 200.0], [100.0, 200.0]) == 0.0

    def test_known_value(self) -> None:
        assert mape([100.0, 200.0], [110.0, 180.0]) == 10.0

    def test_zero_actual_row_dropped(self) -> None:
        assert mape([0.0, 100.0], [5.0, 110.0]) == 10.0

    def test_all_zero_actual_is_nan(self) -> None:
        assert math.isnan(mape([0.0, 0.0], [1.0, 2.0]))


class TestCoverage:
    def test_full(self) -> None:
        assert coverage([1.0, 2.0], [0.0, 1.0], [2.0, 3.0]) == 100.0

    def test_none(self) -> None:
        assert coverage([5.0, 6.0], [0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_half(self) -> None:
        assert coverage([1.0, 9.0], [0.0, 0.0], [2.0, 2.0]) == 50.0

    def test_empty_is_nan(self) -> None:
        assert math.isnan(coverage([], [], []))


class TestBacktestPair:
    def test_skips_when_insufficient_history(self) -> None:
        df = _make_gold_df("USD_EUR", n=20)
        result = backtest_pair(
            df,
            "USD_EUR",
            forecaster=_perfect_forecaster,
            backtest_horizon=10,
            min_training_rows=30,
        )
        assert result is None

    def test_perfect_forecaster_scores(self) -> None:
        df = _make_gold_df("USD_EUR", n=50, rate=1.5)
        result = backtest_pair(
            df,
            "USD_EUR",
            forecaster=_perfect_forecaster,
            backtest_horizon=10,
            min_training_rows=30,
        )
        assert result is not None
        assert result["n_train"] == 40
        assert result["n_eval"] == 10
        assert result["mae"] == 0.0
        assert result["mape"] == 0.0
        assert result["coverage"] == 100.0
        assert result["currency_pair"] == "USD_EUR"

    def test_biased_forecaster_scores(self) -> None:
        df = _make_gold_df("USD_EUR", n=50, rate=100.0)
        result = backtest_pair(
            df,
            "USD_EUR",
            forecaster=_biased_forecaster,
            backtest_horizon=10,
            min_training_rows=30,
        )
        assert result is not None
        assert result["mae"] == 10.0
        assert result["mape"] == 10.0
        assert result["coverage"] == 0.0
