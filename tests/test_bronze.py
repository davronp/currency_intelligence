"""tests/test_bronze.py

Unit tests for src/bronze/raw_to_bronze.py.
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

import pytest
from pyspark.sql import Row

from src.bronze.raw_to_bronze import (
    _parse_json_file,
    load_raw_files,
    transform_to_bronze,
)
from src.utils.schema import BRONZE_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


def _write_rates_json(directory: Path, base: str, rates: dict, run_date: date) -> Path:
    """Helper to write a minimal rates JSON file."""
    date_dir = directory / run_date.isoformat()
    date_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "base": base,
        "date": run_date.isoformat(),
        "rates": rates,
        "fetched_at": f"{run_date.isoformat()}T10:00:00+00:00",
        "source": "https://mock.api",
    }
    out = date_dir / f"rates_{base}.json"
    out.write_text(json.dumps(payload))
    return out


class TestParseJsonFile:
    def test_produces_one_row_per_currency(self, tmp_path):
        jf = _write_rates_json(
            tmp_path,
            "USD",
            {"EUR": 0.92, "SEK": 10.45, "UZS": 12650.0},
            date(2024, 1, 15),
        )
        rows = _parse_json_file(jf)
        assert len(rows) == 3

    def test_row_has_required_keys(self, tmp_path):
        jf = _write_rates_json(tmp_path, "USD", {"EUR": 0.92}, date(2024, 1, 15))
        rows = _parse_json_file(jf)
        required = {
            "ingestion_date",
            "base_currency",
            "target_currency",
            "rate",
            "source",
            "ingested_at",
        }
        assert required.issubset(rows[0].keys())

    def test_rate_value_is_float(self, tmp_path):
        jf = _write_rates_json(tmp_path, "USD", {"EUR": 0.92}, date(2024, 1, 15))
        rows = _parse_json_file(jf)
        assert isinstance(rows[0]["rate"], float)

    def test_base_currency_set_correctly(self, tmp_path):
        jf = _write_rates_json(tmp_path, "USD", {"SEK": 10.45}, date(2024, 1, 15))
        rows = _parse_json_file(jf)
        assert rows[0]["base_currency"] == "USD"

    def test_empty_rates_produces_no_rows(self, tmp_path):
        jf = _write_rates_json(tmp_path, "USD", {}, date(2024, 1, 15))
        rows = _parse_json_file(jf)
        assert rows == []


class TestLoadRawFiles:
    def test_raises_when_date_dir_missing(self, spark, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_files(spark, tmp_path, run_date=date(2099, 1, 1))

    def test_raises_when_no_json_files(self, spark, tmp_path):
        empty_dir = tmp_path / "2024-01-15"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No JSON rate files"):
            load_raw_files(spark, tmp_path, run_date=date(2024, 1, 15))

    def test_loads_multiple_json_files(self, spark, tmp_path):
        run_date = date(2024, 1, 15)
        _write_rates_json(tmp_path, "USD", {"EUR": 0.92, "SEK": 10.45}, run_date)
        df = load_raw_files(spark, tmp_path, run_date=run_date)
        assert df.count() == 2

    def test_columns_match_bronze_schema_names(self, spark, tmp_path):
        run_date = date(2024, 1, 15)
        _write_rates_json(tmp_path, "USD", {"EUR": 0.92}, run_date)
        df = load_raw_files(spark, tmp_path, run_date=run_date)
        expected_cols = {f.name for f in BRONZE_SCHEMA.fields}
        assert expected_cols.issubset(set(df.columns))


class TestTransformToBronze:
    def _make_df(self, spark, rows):
        return spark.createDataFrame(rows)

    def test_deduplication(self, spark):
        rows = [
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="EUR",
                rate=0.92,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            ),
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="EUR",
                rate=0.93,
                source="api",
                ingested_at="2024-01-15T11:00:00",
            ),
        ]
        df = spark.createDataFrame(rows)
        result = transform_to_bronze(df)
        assert result.count() == 1
        # Should keep the later ingestion
        assert result.collect()[0]["rate"] == 0.93

    def test_schema_enforced(self, spark):
        rows = [
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="SEK",
                rate=10.45,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            )
        ]
        df = spark.createDataFrame(rows)
        result = transform_to_bronze(df)
        result_schema_names = {f.name for f in result.schema.fields}
        expected_schema_names = {f.name for f in BRONZE_SCHEMA.fields}
        assert result_schema_names == expected_schema_names

    def test_ingested_at_is_timestamp(self, spark):
        from pyspark.sql.types import TimestampType

        rows = [
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="EUR",
                rate=0.92,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            )
        ]
        df = spark.createDataFrame(rows)
        result = transform_to_bronze(df)
        ingested_at_type = result.schema["ingested_at"].dataType
        assert isinstance(ingested_at_type, TimestampType)
