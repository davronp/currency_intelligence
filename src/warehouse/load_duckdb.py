"""src/warehouse/load_duckdb.py.

Warehouse layer: load curated gold Parquet tables and ML forecasts
into DuckDB for fast analytical queries.

DuckDB can read Parquet natively via ``read_parquet()`` so data is
never duplicated on disk - the warehouse is a logical view layer
backed by the Parquet lake files.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

try:
    import duckdb  # type: ignore[import-untyped]

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("duckdb not installed - warehouse layer unavailable")


def _assert_duckdb() -> None:
    if not DUCKDB_AVAILABLE:
        msg = "duckdb is not installed.  Run: pip install duckdb"
        raise ImportError(msg)


def get_connection(
    db_path: Path,
    retries: int = 5,
    retry_delay: float = 2.0,
) -> duckdb.DuckDBPyConnection:
    """Open (or create) a DuckDB database file, retrying on lock contention.

    Parameters
    ----------
    db_path:
        File-system path to the ``.duckdb`` file.
    retries:
        Number of attempts before giving up.
    retry_delay:
        Seconds to wait between attempts.

    Returns
    -------
    duckdb.DuckDBPyConnection

    """
    _assert_duckdb()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, retries + 1):
        try:
            conn = duckdb.connect(str(db_path))
            logger.info("DuckDB connected: %s", db_path)
        except duckdb.IOException as exc:
            last_exc = exc
            logger.warning(
                "DuckDB lock contention on %s (attempt %d/%d), retrying in %.0fs...",
                db_path.name,
                attempt,
                retries,
                retry_delay,
            )
            time.sleep(retry_delay)
        else:
            return conn

    msg = (
        f"Could not acquire DuckDB lock on {db_path} after {retries} attempts. "
        "Another process may have it open. Close it and retry."
    )
    raise RuntimeError(msg) from last_exc


def load_parquet_as_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    parquet_path: Path,
    *,
    hive_partitioned: bool = False,
) -> int:
    """(Re)create a DuckDB table backed by a Parquet dataset.

    The table is always **replaced** so the operation is idempotent.

    Parameters
    ----------
    conn:
        Open DuckDB connection.
    table_name:
        Destination table name.
    parquet_path:
        Directory or file path with Parquet data.
    hive_partitioned:
        When True, uses a ``**/*.parquet`` glob with
        ``hive_partitioning=true`` (Bronze layer).
        When False, reads all ``.parquet`` files directly (Silver/Gold).

    Returns
    -------
    int
        Row count of the loaded table.

    """
    if not parquet_path.exists():
        logger.warning("Parquet path does not exist, skipping: %s", parquet_path)
        return 0

    if parquet_path.is_file():
        glob = str(parquet_path)
        hive_flag = "false"
    elif hive_partitioned:
        glob = f"{parquet_path}/**/*.parquet"
        hive_flag = "true"
    else:
        glob = f"{parquet_path}/*.parquet"
        hive_flag = "false"

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * FROM read_parquet('{glob}', hive_partitioning={hive_flag});
    """
    conn.execute(sql)
    row_count: int = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info("Loaded table '%s': %d rows", table_name, row_count)
    return row_count


def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True if *table_name* exists in the current DuckDB catalog."""
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
        [table_name],
    ).fetchone()
    return result[0] > 0


def create_analytical_views(conn: duckdb.DuckDBPyConnection) -> None:
    """Create convenience views on top of the raw tables.

    Views
    -----
    v_latest_rates:
        Most recent rate per currency pair (always created).
    v_forecast_summary:
        Next-30-day forecast with uncertainty range.
        Only created when the ``ml_forecasts`` table exists.
    """
    conn.execute("""
        CREATE OR REPLACE VIEW v_latest_rates AS
        SELECT
            currency_pair,
            date,
            rate,
            ma_7,
            ma_30,
            ma_90,
            volatility_30,
            rate_z_score
        FROM gold_exchange_rates
        WHERE date = (SELECT MAX(date) FROM gold_exchange_rates)
        ORDER BY currency_pair;
    """)
    logger.info("Analytical view created: v_latest_rates")

    if _table_exists(conn, "ml_forecasts"):
        conn.execute("""
            CREATE OR REPLACE VIEW v_forecast_summary AS
            SELECT
                f.currency_pair,
                f.forecast_date,
                f.yhat         AS forecast_rate,
                f.yhat_lower   AS lower_bound,
                f.yhat_upper   AS upper_bound,
                f.yhat_upper - f.yhat_lower AS uncertainty_range,
                f.model_trained_at
            FROM ml_forecasts f
            ORDER BY f.currency_pair, f.forecast_date;
        """)
        logger.info("Analytical view created: v_forecast_summary")
    else:
        logger.warning(
            "Skipping v_forecast_summary - ml_forecasts table not loaded "
            "(pipeline needs ≥%d days of data to train Prophet models)",
            30,
        )


def run_warehouse(
    db_path: Path,
    gold_dir: Path,
    forecasts_dir: Path,
    table_names: dict[str, str] | None = None,
) -> None:
    """Full warehouse pipeline: connect -> load tables -> create views -> close.

    Parameters
    ----------
    db_path:
        DuckDB file path.
    gold_dir:
        Gold Parquet lake directory.
    forecasts_dir:
        Forecasts Parquet directory.
    table_names:
        Optional override for table name mapping.
        Defaults to ``{"exchange_rates": "gold_exchange_rates",
                       "forecasts": "ml_forecasts"}``.

    """
    _assert_duckdb()
    logger.info("Starting warehouse load pipeline")

    table_names = table_names or {
        "exchange_rates": "gold_exchange_rates",
        "forecasts": "ml_forecasts",
    }

    conn = get_connection(db_path)
    try:
        load_parquet_as_table(
            conn,
            table_names["exchange_rates"],
            gold_dir,
            hive_partitioned=False,  # Gold is a single coalesced file
        )

        forecast_file = forecasts_dir / "forecasts.parquet"
        if forecast_file.exists():
            load_parquet_as_table(
                conn,
                table_names["forecasts"],
                forecast_file,
                hive_partitioned=False,
            )
        else:
            logger.warning("Forecasts file not found: %s", forecast_file)

        create_analytical_views(conn)
        logger.info("Warehouse pipeline complete")
    finally:
        conn.close()
        logger.info("DuckDB connection closed")
