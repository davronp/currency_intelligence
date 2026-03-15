"""tests/test_gold.py.

Unit tests for src/gold/silver_to_gold.py.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F

from src.gold.silver_to_gold import (
    add_moving_averages,
    add_rate_z_score,
    add_volatility,
    transform_to_gold,
)


def _silver_df(spark, n_days: int = 60, pair: str = "USD_EUR", base_rate: float = 0.92):
    """Synthetic silver DataFrame with a sine-wave rate over *n_days*."""
    start = date(2024, 1, 1)
    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        rate = base_rate + 0.02 * math.sin(i / 5.0)
        prev_rate = base_rate + 0.02 * math.sin((i - 1) / 5.0)
        daily_return = (rate / prev_rate) - 1.0 if i > 0 else None
        rows.append(
            Row(
                date=d.isoformat(),
                currency_pair=pair,
                base_currency=pair.split("_", maxsplit=1)[0],
                target_currency=pair.split("_")[1],
                rate=float(rate),
                prev_rate=float(prev_rate) if i > 0 else None,
                daily_return=float(daily_return) if daily_return is not None else None,
                processed_at="2024-01-01T00:00:00",
            )
        )
    df = spark.createDataFrame(rows)
    df = df.withColumn("date", F.to_date(F.col("date")))
    return df.withColumn("processed_at", F.to_timestamp(F.col("processed_at")))


def _three_pairs_df(spark, n_days: int = 60):
    """Silver DataFrame with all three active pairs: USD_EUR, USD_SEK, USD_GBP."""
    return (
        _silver_df(spark, n_days, "USD_EUR", 0.92)
        .unionByName(_silver_df(spark, n_days, "USD_SEK", 10.45))
        .unionByName(_silver_df(spark, n_days, "USD_GBP", 0.79))
    )


class TestAddMovingAverages:
    def test_ma_columns_added(self, spark):
        df = add_moving_averages(_silver_df(spark), windows=[7, 30, 90])
        assert {"ma_7", "ma_30", "ma_90"}.issubset(df.columns)

    def test_ma7_non_null_after_7_rows(self, spark):
        df = add_moving_averages(_silver_df(spark, n_days=20), windows=[7]).orderBy("date")
        rows = df.collect()
        for i, row in enumerate(rows):
            if i >= 6:
                assert row["ma_7"] is not None

    def test_constant_rate_ma_equals_rate(self, spark):
        from datetime import date, timedelta

        start = date(2024, 1, 1)
        rows = [
            Row(
                date=(start + timedelta(days=i)).isoformat(),
                currency_pair="USD_EUR",
                base_currency="USD",
                target_currency="EUR",
                rate=1.0,
                prev_rate=1.0,
                daily_return=0.0,
                processed_at="2024-01-01T00:00:00",
            )
            for i in range(10)
        ]
        df = spark.createDataFrame(rows)
        df = df.withColumn("date", F.to_date(F.col("date")))
        df = df.withColumn("processed_at", F.to_timestamp(F.col("processed_at")))
        result = add_moving_averages(df, windows=[7]).orderBy("date")
        assert result.collect()[-1]["ma_7"] == pytest.approx(1.0, rel=1e-4)

    def test_all_three_pairs_get_ma(self, spark):
        df = add_moving_averages(_three_pairs_df(spark), windows=[7])
        pairs = {r.currency_pair for r in df.select("currency_pair").distinct().collect()}
        assert pairs == {"USD_EUR", "USD_SEK", "USD_GBP"}


class TestAddVolatility:
    def test_volatility_column_added(self, spark):
        assert "volatility_30" in add_volatility(_silver_df(spark), window=30).columns

    def test_volatility_non_negative(self, spark):
        df = add_volatility(_silver_df(spark, n_days=60), window=30)
        vols = [r.volatility_30 for r in df.collect() if r.volatility_30 is not None]
        assert all(v >= 0 for v in vols)

    def test_zero_volatility_for_constant_returns(self, spark):
        from datetime import date, timedelta

        start = date(2024, 1, 1)
        rows = [
            Row(
                date=(start + timedelta(days=i)).isoformat(),
                currency_pair="USD_SEK",
                base_currency="USD",
                target_currency="SEK",
                rate=10.45,
                prev_rate=10.45,
                daily_return=0.001,
                processed_at="2024-01-01T00:00:00",
            )
            for i in range(35)
        ]
        df = spark.createDataFrame(rows)
        df = df.withColumn("date", F.to_date(F.col("date")))
        df = df.withColumn("processed_at", F.to_timestamp(F.col("processed_at")))
        result = add_volatility(df, window=30).orderBy("date")
        assert result.collect()[-1]["volatility_30"] == pytest.approx(0.0, abs=1e-10)

    def test_three_pairs_independent_volatility(self, spark):
        df = add_volatility(_three_pairs_df(spark, n_days=60), window=30)
        for pair in ["USD_EUR", "USD_SEK", "USD_GBP"]:
            vols = [
                r.volatility_30
                for r in df.filter(F.col("currency_pair") == pair).collect()
                if r.volatility_30 is not None
            ]
            assert len(vols) > 0


class TestAddRateZScore:
    def test_z_score_column_added(self, spark):
        assert "rate_z_score" in add_rate_z_score(_silver_df(spark)).columns

    def test_z_scores_mean_near_zero(self, spark):
        df = add_rate_z_score(_silver_df(spark, n_days=60))
        mean_z = df.agg(F.avg("rate_z_score")).collect()[0][0]
        assert mean_z == pytest.approx(0.0, abs=1e-6)

    def test_all_three_pairs_independent_z_scores(self, spark):
        df = add_rate_z_score(_three_pairs_df(spark, n_days=40))
        for pair in ["USD_EUR", "USD_SEK", "USD_GBP"]:
            mean_z = df.filter(F.col("currency_pair") == pair).agg(F.avg("rate_z_score")).collect()[0][0]
            assert mean_z == pytest.approx(0.0, abs=1e-6), f"{pair} z-score mean off"


class TestTransformToGold:
    def test_output_schema_matches_gold_schema(self, spark):
        from src.utils.schema import GOLD_SCHEMA

        result = transform_to_gold(_silver_df(spark, n_days=100))
        assert set(result.columns) == {f.name for f in GOLD_SCHEMA.fields}

    def test_row_count_preserved(self, spark):
        df = _silver_df(spark, n_days=60)
        assert transform_to_gold(df).count() == 60

    def test_idempotent(self, spark):
        df = _silver_df(spark, n_days=60)
        assert transform_to_gold(df).count() == transform_to_gold(df).count()

    def test_three_pairs_all_present(self, spark):
        df = _three_pairs_df(spark, n_days=60)
        result = transform_to_gold(df)
        pairs = {r.currency_pair for r in result.select("currency_pair").distinct().collect()}
        assert pairs == {"USD_EUR", "USD_SEK", "USD_GBP"}

    def test_gold_row_count_three_pairs(self, spark):
        """60 days x 3 pairs = 180 rows."""
        df = _three_pairs_df(spark, n_days=60)
        assert transform_to_gold(df).count() == 180
