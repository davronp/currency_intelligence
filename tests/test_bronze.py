"""tests/test_bronze.py.

Unit tests for src/bronze/raw_to_bronze.py.

Covers:
- JSONL parsing (single and multi-date files)
- load_raw_files reading from flat JSONL directory
- transform_to_bronze schema enforcement and deduplication
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING

import pytest
from pyspark.sql import Row

from src.bronze.raw_to_bronze import (
    _parse_jsonl_file,
    load_raw_files,
    transform_to_bronze,
)
from src.utils.schema import BRONZE_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(directory: Path, base: str, records: list[dict]) -> Path:
    """Write a rates_<BASE>.jsonl file with one record per entry in *records*."""
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"rates_{base}.jsonl"
    with open(out, "w") as fh:
        fh.writelines(json.dumps(rec) + "\n" for rec in records)
    return out


def _make_record(d: date, base: str = "USD", rates: dict | None = None) -> dict:
    if rates is None:
        rates = {"EUR": 0.92, "SEK": 10.45, "GBP": 0.79}
    return {
        "base": base,
        "date": d.isoformat(),
        "rates": rates,
        "fetched_at": f"{d.isoformat()}T10:00:00+00:00",
        "source": "https://api.frankfurter.app",
    }


class TestParseJsonlFile:
    def test_produces_one_row_per_currency_per_date(self, tmp_path):
        jf = _write_jsonl(
            tmp_path,
            "USD",
            [
                _make_record(date(2024, 1, 13), rates={"EUR": 0.91, "SEK": 10.40, "GBP": 0.78}),
                _make_record(date(2024, 1, 14), rates={"EUR": 0.92, "SEK": 10.45, "GBP": 0.79}),
            ],
        )
        rows = _parse_jsonl_file(jf)
        # 2 dates x 3 currencies = 6 rows
        assert len(rows) == 6

    def test_row_has_required_keys(self, tmp_path):
        jf = _write_jsonl(tmp_path, "USD", [_make_record(date(2024, 1, 15))])
        rows = _parse_jsonl_file(jf)
        required = {"ingestion_date", "base_currency", "target_currency", "rate", "source", "ingested_at"}
        assert required.issubset(rows[0].keys())

    def test_rate_value_is_float(self, tmp_path):
        jf = _write_jsonl(tmp_path, "USD", [_make_record(date(2024, 1, 15), rates={"EUR": 0.92})])
        rows = _parse_jsonl_file(jf)
        assert isinstance(rows[0]["rate"], float)

    def test_base_currency_set_correctly(self, tmp_path):
        jf = _write_jsonl(tmp_path, "USD", [_make_record(date(2024, 1, 15), rates={"SEK": 10.45})])
        rows = _parse_jsonl_file(jf)
        assert all(r["base_currency"] == "USD" for r in rows)

    def test_empty_rates_produces_no_rows(self, tmp_path):
        jf = _write_jsonl(tmp_path, "USD", [_make_record(date(2024, 1, 15), rates={})])
        rows = _parse_jsonl_file(jf)
        assert rows == []

    def test_malformed_line_skipped(self, tmp_path):
        out = tmp_path / "rates_USD.jsonl"
        valid = '{"base":"USD","date":"2024-01-15","rates":{"EUR":0.92},'
        valid += '"fetched_at":"2024-01-15T10:00:00+00:00","source":"test"}'
        out.write_text(valid + "\nnot valid json\n")
        rows = _parse_jsonl_file(out)
        assert len(rows) == 1

    def test_single_date_three_currencies(self, tmp_path):
        jf = _write_jsonl(
            tmp_path, "USD", [_make_record(date(2024, 1, 15), rates={"EUR": 0.92, "SEK": 10.45, "GBP": 0.79})]
        )
        rows = _parse_jsonl_file(jf)
        assert len(rows) == 3
        targets = {r["target_currency"] for r in rows}
        assert targets == {"EUR", "SEK", "GBP"}


class TestLoadRawFiles:
    def test_raises_when_no_jsonl_files(self, spark, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"No rates_\*.jsonl"):
            load_raw_files(spark, tmp_path)

    def test_loads_single_jsonl_file(self, spark, tmp_path):
        _write_jsonl(
            tmp_path,
            "USD",
            [
                _make_record(date(2024, 1, 13), rates={"EUR": 0.91, "SEK": 10.40}),
                _make_record(date(2024, 1, 14), rates={"EUR": 0.92, "SEK": 10.45}),
            ],
        )
        df = load_raw_files(spark, tmp_path)
        # 2 dates x 2 currencies = 4 rows
        assert df.count() == 4

    def test_columns_match_bronze_schema(self, spark, tmp_path):
        _write_jsonl(tmp_path, "USD", [_make_record(date(2024, 1, 15))])
        df = load_raw_files(spark, tmp_path)
        expected = {f.name for f in BRONZE_SCHEMA.fields}
        assert expected.issubset(set(df.columns))

    def test_raises_on_empty_jsonl(self, spark, tmp_path):
        (tmp_path / "rates_USD.jsonl").write_text("")
        with pytest.raises(ValueError, match="No rows parsed"):
            load_raw_files(spark, tmp_path)

    def test_no_subdirectories_required(self, spark, tmp_path):
        """JSONL files sit directly in raw_dir — no date subdirectories."""
        _write_jsonl(tmp_path, "USD", [_make_record(date(2024, 1, 15))])
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs == []
        df = load_raw_files(spark, tmp_path)
        assert df.count() > 0


class TestTransformToBronze:
    def _raw_df(self, spark, rows=None):
        if rows is None:
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
        return spark.createDataFrame(rows)

    def test_schema_enforced(self, spark):
        df = self._raw_df(spark)
        result = transform_to_bronze(df)
        result_cols = {f.name for f in result.schema.fields}
        expected_cols = {f.name for f in BRONZE_SCHEMA.fields}
        assert result_cols == expected_cols

    def test_deduplication_keeps_latest_ingested_at(self, spark):
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
        result = transform_to_bronze(spark.createDataFrame(rows))
        assert result.count() == 1
        assert result.collect()[0]["rate"] == 0.93

    def test_multiple_dates_preserved(self, spark):
        rows = [
            Row(
                ingestion_date="2024-01-13",
                base_currency="USD",
                target_currency="EUR",
                rate=0.91,
                source="api",
                ingested_at="2024-01-13T10:00:00",
            ),
            Row(
                ingestion_date="2024-01-14",
                base_currency="USD",
                target_currency="EUR",
                rate=0.92,
                source="api",
                ingested_at="2024-01-14T10:00:00",
            ),
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency="EUR",
                rate=0.93,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            ),
        ]
        result = transform_to_bronze(spark.createDataFrame(rows))
        assert result.count() == 3

    def test_ingestion_date_is_date_type(self, spark):
        from pyspark.sql.types import DateType

        df = self._raw_df(spark)
        result = transform_to_bronze(df)
        assert isinstance(result.schema["ingestion_date"].dataType, DateType)

    def test_ingested_at_is_timestamp_type(self, spark):
        from pyspark.sql.types import TimestampType

        df = self._raw_df(spark)
        result = transform_to_bronze(df)
        assert isinstance(result.schema["ingested_at"].dataType, TimestampType)

    def test_all_three_currency_pairs(self, spark):
        rows = [
            Row(
                ingestion_date="2024-01-15",
                base_currency="USD",
                target_currency=t,
                rate=r,
                source="api",
                ingested_at="2024-01-15T10:00:00",
            )
            for t, r in [("EUR", 0.92), ("SEK", 10.45), ("GBP", 0.79)]
        ]
        result = transform_to_bronze(spark.createDataFrame(rows))
        assert result.count() == 3
        targets = {r["target_currency"] for r in result.collect()}
        assert targets == {"EUR", "SEK", "GBP"}
