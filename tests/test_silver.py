"""tests/test_silver.py.

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
    """Return a minimal bronze-shaped DataFrame with EUR/SEK/GBP pairs."""
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
        assert filter_valid_rates(df).count() == 1

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
        assert filter_valid_rates(df, min_rate=1e-6).count() == 0

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
        assert filter_valid_rates(df).count() == 0

    def test_keeps_all_valid_pairs(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency=t,
                    rate=r,
                    source="api",
                    ingested_at="2024-01-15",
                )
                for t, r in [("EUR", 0.92), ("SEK", 10.45), ("GBP", 0.79)]
            ]
        )
        assert filter_valid_rates(df).count() == 3


class TestAddCurrencyPair:
    def test_column_added(self, spark):
        assert "currency_pair" in add_currency_pair(_bronze_rows(spark)).columns

    def test_pair_format_usd_eur(self, spark):
        df = add_currency_pair(_bronze_rows(spark))
        pairs = {r.currency_pair for r in df.select("currency_pair").collect()}
        assert "USD_EUR" in pairs

    def test_pair_format_usd_sek(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="SEK",
                    rate=10.45,
                    source="api",
                    ingested_at="2024-01-15",
                )
            ]
        )
        result = add_currency_pair(df)
        assert result.collect()[0]["currency_pair"] == "USD_SEK"

    def test_pair_format_usd_gbp(self, spark):
        df = spark.createDataFrame(
            [
                Row(
                    ingestion_date="2024-01-15",
                    base_currency="USD",
                    target_currency="GBP",
                    rate=0.79,
                    source="api",
                    ingested_at="2024-01-15",
                )
            ]
        )
        result = add_currency_pair(df)
        assert result.collect()[0]["currency_pair"] == "USD_GBP"

    def test_row_count_unchanged(self, spark):
        df = _bronze_rows(spark)
        assert add_currency_pair(df).count() == df.count()


class TestAddDateColumn:
    def test_date_column_exists(self, spark):
        assert "date" in add_date_column(_bronze_rows(spark)).columns

    def test_ingestion_date_removed(self, spark):
        assert "ingestion_date" not in add_date_column(_bronze_rows(spark)).columns


class TestComputeDailyReturns:
    def _prepare(self, spark):
        df = _bronze_rows(spark)
        df = add_currency_pair(df)
        df = add_date_column(df)
        return df.withColumn("date", F.to_date(F.col("date")))

    def test_prev_rate_column_added(self, spark):
        assert "prev_rate" in compute_daily_returns(self._prepare(spark)).columns

    def test_daily_return_column_added(self, spark):
        assert "daily_return" in compute_daily_returns(self._prepare(spark)).columns

    def test_first_row_return_is_null(self, spark):
        result = compute_daily_returns(self._prepare(spark)).orderBy("date")
        assert result.collect()[0]["daily_return"] is None

    def test_return_value_correct(self, spark):
        """0.90 -> 0.92: return = (0.92/0.90) - 1."""
        rows = compute_daily_returns(self._prepare(spark)).orderBy("date").collect()
        assert rows[1]["daily_return"] == pytest.approx((0.92 / 0.90) - 1, rel=1e-4)

    def test_three_pairs_independent(self, spark):
        """EUR, SEK and GBP daily returns should be computed independently."""
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
            Row(
                date="2024-01-13",
                currency_pair="USD_GBP",
                base_currency="USD",
                target_currency="GBP",
                rate=0.78,
                source="api",
                ingested_at="2024-01-13",
            ),
            Row(
                date="2024-01-14",
                currency_pair="USD_GBP",
                base_currency="USD",
                target_currency="GBP",
                rate=0.79,
                source="api",
                ingested_at="2024-01-14",
            ),
        ]
        df = spark.createDataFrame(rows).withColumn("date", F.to_date(F.col("date")))
        result = {(r.currency_pair, str(r.date)): r.daily_return for r in compute_daily_returns(df).collect()}
        # First day per pair always null
        for pair in ["USD_EUR", "USD_SEK", "USD_GBP"]:
            assert result[(pair, "2024-01-13")] is None
        # Second day returns are independent
        assert result[("USD_EUR", "2024-01-14")] == pytest.approx((0.92 / 0.90) - 1, rel=1e-4)
        assert result[("USD_SEK", "2024-01-14")] == pytest.approx((10.5 / 10.0) - 1, rel=1e-4)
        assert result[("USD_GBP", "2024-01-14")] == pytest.approx((0.79 / 0.78) - 1, rel=1e-4)


class TestTransformToSilver:
    def test_output_schema_matches_silver_schema(self, spark):
        from src.utils.schema import SILVER_SCHEMA

        df = _bronze_rows(spark)
        df = df.withColumn("ingestion_date", F.to_date(F.col("ingestion_date")))
        df = df.withColumn("ingested_at", F.to_timestamp(F.col("ingested_at")))
        result = transform_to_silver(df)
        assert set(result.columns) == {f.name for f in SILVER_SCHEMA.fields}

    def test_invalid_rates_excluded(self, spark):
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
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="GBP",
                rate=0.79,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            ),
        ]
        df = spark.createDataFrame(rows)
        df = df.withColumn("ingestion_date", F.to_date(F.col("ingestion_date")))
        df = df.withColumn("ingested_at", F.to_timestamp(F.col("ingested_at")))
        result = transform_to_silver(df)
        assert all(r.rate > 0 for r in result.collect())
        assert result.count() == 2

    def test_currency_pair_column_present(self, spark):
        df = _bronze_rows(spark)
        df = df.withColumn("ingestion_date", F.to_date(F.col("ingestion_date")))
        df = df.withColumn("ingested_at", F.to_timestamp(F.col("ingested_at")))
        result = transform_to_silver(df)
        assert "currency_pair" in result.columns
        pairs = {r.currency_pair for r in result.collect()}
        assert "USD_EUR" in pairs
