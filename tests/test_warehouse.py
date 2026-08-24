"""Tests for the DuckDB warehouse layer.

These use DuckDB directly (a base dependency) and need no Spark/JVM, so
they run anywhere. Assertions use ``fetchall()`` (non-optional) rather
than ``fetchone()[0]`` to stay clean under the type checker.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import duckdb
import pandas as pd

from src.warehouse.load_duckdb import (
    create_analytical_views,
    get_connection,
    load_parquet_as_table,
    run_warehouse,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_gold(gold_dir: Path) -> None:
    """Write a tiny two-pair, two-day gold dataset as a Parquet part file."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "currency_pair": ["USD_EUR", "USD_EUR", "USD_GBP", "USD_GBP"],
            "rate": [0.90, 0.91, 0.79, 0.80],
            "ma_7": [0.90, 0.905, 0.79, 0.795],
            "ma_30": [0.90, 0.905, 0.79, 0.795],
            "ma_90": [0.90, 0.905, 0.79, 0.795],
            "volatility_30": [0.01, 0.01, 0.02, 0.02],
            "rate_z_score": [0.1, 0.2, -0.1, -0.2],
        }
    ).to_parquet(gold_dir / "part-0.parquet", index=False)


def _write_forecasts(forecasts_dir: Path) -> None:
    """Write a small forecasts.parquet file."""
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    pd.DataFrame(
        {
            "currency_pair": ["USD_EUR", "USD_EUR"],
            "forecast_date": pd.to_datetime(["2026-01-03", "2026-01-04"]),
            "yhat": [0.92, 0.93],
            "yhat_lower": [0.90, 0.91],
            "yhat_upper": [0.94, 0.95],
            "model_trained_at": [now, now],
            "training_rows": [100, 100],
        }
    ).to_parquet(forecasts_dir / "forecasts.parquet", index=False)


class TestGetConnection:
    def test_creates_db_file_and_parents(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "test.duckdb"
        conn = get_connection(db)
        try:
            assert db.exists()
        finally:
            conn.close()


class TestLoadParquetAsTable:
    def test_missing_path_returns_zero(self, tmp_path: Path) -> None:
        conn = duckdb.connect()
        try:
            assert load_parquet_as_table(conn, "t", tmp_path / "nope") == 0
        finally:
            conn.close()

    def test_loads_directory(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold"
        _write_gold(gold)
        conn = duckdb.connect()
        try:
            assert load_parquet_as_table(conn, "gold_exchange_rates", gold) == 4
        finally:
            conn.close()

    def test_loads_single_file(self, tmp_path: Path) -> None:
        fdir = tmp_path / "forecasts"
        _write_forecasts(fdir)
        conn = duckdb.connect()
        try:
            assert load_parquet_as_table(conn, "ml_forecasts", fdir / "forecasts.parquet") == 2
        finally:
            conn.close()


class TestCreateAnalyticalViews:
    def test_latest_rates_uses_max_date(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold"
        _write_gold(gold)
        conn = duckdb.connect()
        try:
            load_parquet_as_table(conn, "gold_exchange_rates", gold)
            create_analytical_views(conn)
            rows = conn.execute(
                "SELECT currency_pair, rate FROM v_latest_rates ORDER BY currency_pair"
            ).fetchall()
            # Latest day (2026-01-02) rates for each pair.
            assert len(rows) == 2
            assert {round(r[1], 2) for r in rows} == {0.91, 0.80}
        finally:
            conn.close()

    def test_forecast_view_skipped_without_table(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold"
        _write_gold(gold)
        conn = duckdb.connect()
        try:
            load_parquet_as_table(conn, "gold_exchange_rates", gold)
            create_analytical_views(conn)
            present = conn.execute(
                "SELECT COUNT(*) FROM information_schema.views WHERE table_name = 'v_forecast_summary'"
            ).fetchall()
            assert present[0][0] == 0
        finally:
            conn.close()


class TestRunWarehouse:
    def test_end_to_end_builds_tables_and_views(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold"
        forecasts = tmp_path / "forecasts"
        _write_gold(gold)
        _write_forecasts(forecasts)
        db = tmp_path / "wh.duckdb"

        run_warehouse(db, gold, forecasts)

        conn = duckdb.connect(str(db), read_only=True)
        try:
            assert conn.execute("SELECT COUNT(*) FROM gold_exchange_rates").fetchall()[0][0] == 4
            assert conn.execute("SELECT COUNT(*) FROM ml_forecasts").fetchall()[0][0] == 2
            assert conn.execute("SELECT COUNT(*) FROM v_latest_rates").fetchall()[0][0] == 2
            assert conn.execute("SELECT COUNT(*) FROM v_forecast_summary").fetchall()[0][0] == 2
        finally:
            conn.close()

    def test_runs_without_forecasts(self, tmp_path: Path) -> None:
        gold = tmp_path / "gold"
        _write_gold(gold)
        forecasts = tmp_path / "forecasts"
        forecasts.mkdir()
        db = tmp_path / "wh.duckdb"

        run_warehouse(db, gold, forecasts)

        conn = duckdb.connect(str(db), read_only=True)
        try:
            assert conn.execute("SELECT COUNT(*) FROM v_latest_rates").fetchall()[0][0] == 2
        finally:
            conn.close()
