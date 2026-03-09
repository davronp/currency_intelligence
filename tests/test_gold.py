"""tests/test_gold.py

Unit tests for src/gold/silver_to_gold.py.
"""

from __future__ import annotations

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
    """Generate a synthetic silver DataFrame with ``n_days`` rows."""
    import math
    from datetime import date, timedelta

    rows = []
    start = date(2024, 1, 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        # Slight sine wave around base_rate
        rate = base_rate + 0.02 * math.sin(i / 5.0)
        daily_return = (
            (0.02 * math.cos(i / 5.0)) / (base_rate + 0.02 * math.sin((i - 1) / 5.0)) if i > 0 else None
        )
        rows.append(
            Row(
                date=d.isoformat(),
                currency_pair=pair,
                base_currency=pair.split("_")[0],
                target_currency=pair.split("_")[1],
                rate=float(rate),
                prev_rate=float(rate - 0.001) if i > 0 else None,
                daily_return=float(daily_return) if daily_return is not None else None,
                processed_at="2024-01-01T00:00:00",
            )
        )

    df = spark.createDataFrame(rows)
    df = df.withColumn("date", F.to_date(F.col("date")))
    return df.withColumn("processed_at", F.to_timestamp(F.col("processed_at")))


class TestAddMovingAverages:
    def test_ma_columns_added(self, spark):
        df = _silver_df(spark)
        result = add_moving_averages(df, windows=[7, 30])
        assert "ma_7" in result.columns
        assert "ma_30" in result.columns

    def test_ma_not_null_after_window_size(self, spark):
        """ma_7 should be non-null for all rows once 7 rows have been seen."""
        df = _silver_df(spark, n_days=20)
        result = add_moving_averages(df, windows=[7]).orderBy("date")
        rows = result.collect()
        # Row index 6 (7th row) onwards should have a non-null ma_7
        for i, row in enumerate(rows):
            if i >= 6:
                assert row["ma_7"] is not None, f"ma_7 should be non-null at row {i}"

    def test_ma7_is_average_of_last_7(self, spark):
        """For a constant rate, ma_7 should equal the rate itself."""
        rows = [
            Row(
                date=f"2024-01-{str(i + 1).zfill(2)}",
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
        # All ma_7 should be 1.0 after enough rows
        last_ma = result.collect()[-1]["ma_7"]
        assert last_ma == pytest.approx(1.0, rel=1e-4)


class TestAddVolatility:
    def test_volatility_column_added(self, spark):
        df = _silver_df(spark)
        result = add_volatility(df, window=30)
        assert "volatility_30" in result.columns

    def test_volatility_non_negative(self, spark):
        """Volatility (std dev) must always be ≥ 0."""
        df = _silver_df(spark, n_days=60)
        result = add_volatility(df, window=30)
        vols = [r.volatility_30 for r in result.collect() if r.volatility_30 is not None]
        assert all(v >= 0 for v in vols)

    def test_zero_volatility_for_constant_returns(self, spark):
        """When daily_return is always the same, std dev = 0."""
        rows = [
            Row(
                date=f"2024-01-{str(i + 1).zfill(2)}",
                currency_pair="USD_EUR",
                base_currency="USD",
                target_currency="EUR",
                rate=1.0,
                prev_rate=1.0,
                daily_return=0.001,
                processed_at="2024-01-01T00:00:00",
            )
            for i in range(35)
        ]
        df = spark.createDataFrame(rows)
        df = df.withColumn("date", F.to_date(F.col("date")))
        df = df.withColumn("processed_at", F.to_timestamp(F.col("processed_at")))
        result = add_volatility(df, window=30).orderBy("date")
        last_vol = result.collect()[-1]["volatility_30"]
        assert last_vol == pytest.approx(0.0, abs=1e-10)


class TestAddRateZScore:
    def test_z_score_column_added(self, spark):
        df = _silver_df(spark)
        result = add_rate_z_score(df)
        assert "rate_z_score" in result.columns

    def test_z_scores_have_zero_mean(self, spark):
        """z-scores should average to ~0 across a pair."""
        df = _silver_df(spark, n_days=60)
        result = add_rate_z_score(df)
        mean_z = result.filter(F.col("currency_pair") == "USD_EUR").agg(F.avg("rate_z_score")).collect()[0][0]
        assert mean_z == pytest.approx(0.0, abs=1e-6)

    def test_multiple_pairs_independent_z_scores(self, spark):
        """Each pair should have an independent z-score mean of ~0."""
        df_eur = _silver_df(spark, n_days=40, pair="USD_EUR", base_rate=0.92)
        df_sek = _silver_df(spark, n_days=40, pair="USD_SEK", base_rate=10.5)
        df = df_eur.unionByName(df_sek)
        result = add_rate_z_score(df, partition_cols=["currency_pair"])

        for pair in ["USD_EUR", "USD_SEK"]:
            mean_z = result.filter(F.col("currency_pair") == pair).agg(F.avg("rate_z_score")).collect()[0][0]
            assert mean_z == pytest.approx(0.0, abs=1e-6), f"Pair {pair} z-score mean off"


class TestTransformToGold:
    def test_output_schema_matches_gold_schema(self, spark):
        from src.utils.schema import GOLD_SCHEMA

        df = _silver_df(spark, n_days=100)
        result = transform_to_gold(df, rolling_windows=[7, 30, 90], volatility_window=30)
        expected = {f.name for f in GOLD_SCHEMA.fields}
        actual = set(result.columns)
        assert actual == expected

    def test_idempotent_deduplication(self, spark):
        """Running transform_to_gold twice on the same data yields same row count."""
        df = _silver_df(spark, n_days=60)
        result1 = transform_to_gold(df).count()
        result2 = transform_to_gold(df).count()
        assert result1 == result2

    def test_row_count_preserved(self, spark):
        """Gold should have the same row count as silver (1 row per date per pair)."""
        df = _silver_df(spark, n_days=60)
        result = transform_to_gold(df)
        assert result.count() == 60
