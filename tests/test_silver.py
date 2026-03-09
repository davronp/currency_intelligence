"""tests/test_silver.py

Unit tests for src/silver/bronze_to_silver.py.
"""

from __future__ import annotations

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as F

from src.silver.bronze_to_silver import (
    add_currency_pair,
    add_date_column,
    compute_daily_returns,
    filter_valid_rates,
    transform_to_silver,
)


def _bronze_rows(spark, overrides: list[dict] | None = None):
    """Return a minimal bronze-shaped DataFrame."""
    defaults = [
        {
            "ingestion_date": "2024-01-13",
            "base_currency": "USD",
            "target_currency": "EUR",
            "rate": 0.90,
            "source": "api",
            "ingested_at": "2024-01-13T10:00:00",
        },
        {
            "ingestion_date": "2024-01-14",
            "base_currency": "USD",
            "target_currency": "EUR",
            "rate": 0.92,
            "source": "api",
            "ingested_at": "2024-01-14T10:00:00",
        },
        {
            "ingestion_date": "2024-01-15",
            "base_currency": "USD",
            "target_currency": "EUR",
            "rate": 0.91,
            "source": "api",
            "ingested_at": "2024-01-15T10:00:00",
        },
    ]
    rows = overrides or defaults
    return spark.createDataFrame([Row(**r) for r in rows])


class TestFilterValidRates:
    def test_removes_null_rates(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="EUR",
                    rate=None,
                    source="api",
                    ingested_at="2024-01-15",
                ),
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="SEK",
                    rate=10.45,
                    source="api",
                    ingested_at="2024-01-15",
                ),
            ]
        )
        result = filter_valid_rates(df)
        assert result.count() == 1

    def test_removes_zero_rate(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="EUR",
                    rate=0.0,
                    source="api",
                    ingested_at="2024-01-15",
                ),
            ]
        )
        result = filter_valid_rates(df, min_rate=1e-6)
        assert result.count() == 0

    def test_removes_negative_rate(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="EUR",
                    rate=-1.5,
                    source="api",
                    ingested_at="2024-01-15",
                ),
            ]
        )
        result = filter_valid_rates(df)
        assert result.count() == 0

    def test_keeps_valid_rates(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="EUR",
                    rate=0.92,
                    source="api",
                    ingested_at="2024-01-15",
                ),
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="UZS",
                    rate=12650.0,
                    source="api",
                    ingested_at="2024-01-15",
                ),
            ]
        )
        result = filter_valid_rates(df)
        assert result.count() == 2


class TestAddCurrencyPair:
    def test_column_added(self, spark):
        df = _bronze_rows(spark)
        result = add_currency_pair(df)
        assert "currency_pair" in result.columns

    def test_pair_format(self, spark):
        df = _bronze_rows(spark)
        result = add_currency_pair(df)
        pairs = {r.currency_pair for r in result.select("currency_pair").collect()}
        assert "USD_EUR" in pairs

    def test_pair_count_unchanged(self, spark):
        df = _bronze_rows(spark)
        assert add_currency_pair(df).count() == df.count()


class TestAddDateColumn:
    def test_date_column_exists(self, spark):
        df = _bronze_rows(spark)
        result = add_date_column(df)
        assert "date" in result.columns

    def test_ingestion_date_removed(self, spark):
        df = _bronze_rows(spark)
        result = add_date_column(df)
        assert "ingestion_date" not in result.columns


class TestComputeDailyReturns:
    def _prepare(self, spark):
        df = _bronze_rows(spark)
        df = add_currency_pair(df)
        df = add_date_column(df)
        return df.withColumn("date", F.to_date(F.col("date")))

    def test_prev_rate_column_added(self, spark):
        df = self._prepare(spark)
        result = compute_daily_returns(df)
        assert "prev_rate" in result.columns

    def test_daily_return_column_added(self, spark):
        df = self._prepare(spark)
        result = compute_daily_returns(df)
        assert "daily_return" in result.columns

    def test_first_row_return_is_null(self, spark):
        df = self._prepare(spark)
        result = compute_daily_returns(df).orderBy("date")
        first = result.collect()[0]
        assert first["daily_return"] is None

    def test_return_value_correct(self, spark):
        """Rate goes 0.90 → 0.92 → return = (0.92/0.90) - 1 ≈ 0.0222."""
        df = self._prepare(spark)
        rows = compute_daily_returns(df).orderBy("date").collect()
        expected = (0.92 / 0.90) - 1
        assert rows[1]["daily_return"] == pytest.approx(expected, rel=1e-4)

    def test_multiple_pairs_independent(self, spark):
        """Daily returns should be computed per currency pair independently."""
        rows = [
            Row(
                date="2024-01-13",
                currency_pair="USD_EUR",
                base_currency="USD",
                target_currency="EUR",
                rate=0.90,
                source="api",
                ingested_at="2024-01-13",
            ),
            Row(
                date="2024-01-14",
                currency_pair="USD_EUR",
                base_currency="USD",
                target_currency="EUR",
                rate=0.92,
                source="api",
                ingested_at="2024-01-14",
            ),
            Row(
                date="2024-01-13",
                currency_pair="USD_SEK",
                base_currency="USD",
                target_currency="SEK",
                rate=10.0,
                source="api",
                ingested_at="2024-01-13",
            ),
            Row(
                date="2024-01-14",
                currency_pair="USD_SEK",
                base_currency="USD",
                target_currency="SEK",
                rate=10.5,
                source="api",
                ingested_at="2024-01-14",
            ),
        ]
        df = spark.createDataFrame(rows).withColumn("date", F.to_date(F.col("date")))
        result = compute_daily_returns(df).orderBy("currency_pair", "date")
        collected = {(r.currency_pair, str(r.date)): r.daily_return for r in result.collect()}

        # First day per pair should have null return
        assert collected[("USD_EUR", "2024-01-13")] is None
        assert collected[("USD_SEK", "2024-01-13")] is None

        assert collected[("USD_EUR", "2024-01-14")] == pytest.approx((0.92 / 0.90) - 1, rel=1e-4)
        assert collected[("USD_SEK", "2024-01-14")] == pytest.approx((10.5 / 10.0) - 1, rel=1e-4)


class TestTransformToSilver:
    def test_output_schema_matches_silver_schema(self, spark):
        from src.utils.schema import SILVER_SCHEMA

        df = _bronze_rows(spark)
        df = df.withColumn("ingestion_date", F.to_date(F.col("ingestion_date")))
        df = df.withColumn("ingested_at", F.to_timestamp(F.col("ingested_at")))
        result = transform_to_silver(df)
        expected = {f.name for f in SILVER_SCHEMA.fields}
        actual = set(result.columns)
        assert actual == expected

    def test_output_has_no_invalid_rates(self, spark):
        rows = [
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="EUR",
                rate=-0.5,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            ),
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="SEK",
                rate=10.45,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            ),
        ]
        df = spark.createDataFrame(rows)
        df = df.withColumn("ingestion_date", F.to_date(F.col("ingestion_date")))
        df = df.withColumn("ingested_at", F.to_timestamp(F.col("ingested_at")))
        result = transform_to_silver(df)
        rates = [r.rate for r in result.collect()]
        assert all(r > 0 for r in rates)
