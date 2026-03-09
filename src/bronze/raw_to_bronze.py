"""src/bronze/raw_to_bronze.py

Bronze layer: read raw JSON files, parse into a structured
Spark DataFrame, and write Parquet partitioned by ingestion date.

Idempotency: the output is always written with ``mode="overwrite"``
on the partition directory, so re-runs produce the same result.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.logger import get_logger
from src.utils.schema import BRONZE_SCHEMA
from src.utils.spark_utils import deduplicate, enforce_schema, write_parquet

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def _parse_json_file(json_path: Path) -> list[dict]:
    """Parse a single raw JSON file into a list of flat row dicts.

    Each entry in the output represents one currency pair
    observation (base → target) on a given date.

    Parameters
    ----------
    json_path:
        Path to a raw ``rates_<BASE>.json`` file.

    Returns
    -------
    list[dict]
        Flat row dictionaries compatible with ``BRONZE_SCHEMA``.

    """
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    base = payload["base"]
    raw_date_str = payload.get("date", "")
    ingested_at = payload.get("fetched_at", datetime.now(UTC).isoformat())

    # Parse the date from the API date field or fall back to today
    try:
        ingestion_date = datetime.strptime(raw_date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        ingestion_date = date.today()

    rows = []
    for target, rate in payload.get("rates", {}).items():
        rows.append(
            {
                "ingestion_date": ingestion_date.isoformat(),
                "base_currency": base,
                "target_currency": target,
                "rate": float(rate),
                "source": payload.get("source", "unknown"),
                "ingested_at": ingested_at,
            }
        )

    logger.debug(
        "Parsed %d rows from %s (base=%s, date=%s)",
        len(rows),
        json_path.name,
        base,
        ingestion_date,
    )
    return rows


def load_raw_files(
    spark: SparkSession,
    raw_dir: Path,
    run_date: date | None = None,
) -> DataFrame:
    """Scan the raw directory and load all JSON files for *run_date*.

    Parameters
    ----------
    spark:
        Active SparkSession.
    raw_dir:
        Root of the raw data lake (``data/raw``).
    run_date:
        Restrict loading to files under ``data/raw/<YYYY-MM-DD>/``.
        Defaults to today.

    Returns
    -------
    DataFrame
        Unvalidated bronze-shaped DataFrame.

    """
    run_date = run_date or date.today()
    date_dir = raw_dir / run_date.isoformat()

    if not date_dir.exists():
        msg = f"No raw data directory for date: {date_dir}"
        raise FileNotFoundError(msg)

    json_files = sorted(date_dir.glob("rates_*.json"))
    if not json_files:
        msg = f"No JSON rate files found under: {date_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading %d JSON file(s) from %s", len(json_files), date_dir)

    all_rows: list[dict] = []
    for jf in json_files:
        all_rows.extend(_parse_json_file(jf))

    df = spark.createDataFrame(all_rows)
    logger.info("Loaded %d raw row(s)", df.count())
    return df


def transform_to_bronze(df: DataFrame) -> DataFrame:
    """Apply bronze-level transformations and enforce schema.

    Steps
    -----
    1. Cast columns to their declared types.
    2. Deduplicate on (ingestion_date, base_currency, target_currency).

    Parameters
    ----------
    df:
        Raw DataFrame from :func:`load_raw_files`.

    Returns
    -------
    DataFrame
        Schema-enforced, deduplicated bronze DataFrame.

    """
    df = df.withColumn("ingestion_date", F.to_date(F.col("ingestion_date"))).withColumn(
        "ingested_at", F.to_timestamp(F.col("ingested_at"))
    )

    df = enforce_schema(df, BRONZE_SCHEMA)

    df = deduplicate(
        df,
        partition_cols=["ingestion_date", "base_currency", "target_currency"],
        order_col="ingested_at",
    )

    logger.info("Bronze transformation complete (%d rows)", df.count())
    return df


def write_bronze(df: DataFrame, bronze_dir: Path, run_date: date | None = None) -> None:
    """Persist the bronze DataFrame as Parquet, partitioned by ingestion_date.

    Overwrites the partition for *run_date* only (safe idempotent writes).

    Parameters
    ----------
    df:
        Transformed bronze DataFrame.
    bronze_dir:
        Bronze data lake root (``data/bronze``).
    run_date:
        The partition date to overwrite.

    """
    run_date = run_date or date.today()
    out_path = str(bronze_dir / f"ingestion_date={run_date.isoformat()}")

    # Drop the partition column before writing (it is encoded in the path)
    df_out = df.drop("ingestion_date")

    write_parquet(df_out, out_path, mode="overwrite")
    logger.info("Bronze layer written → %s", out_path)


def run_bronze(
    spark: SparkSession,
    raw_dir: Path,
    bronze_dir: Path,
    run_date: date | None = None,
) -> DataFrame:
    """End-to-end bronze pipeline: load → transform → write.

    Returns the bronze DataFrame for downstream chaining.
    """
    run_date = run_date or date.today()
    logger.info("Starting bronze pipeline for date=%s", run_date)

    df_raw = load_raw_files(spark, raw_dir, run_date)
    df_bronze = transform_to_bronze(df_raw)
    write_bronze(df_bronze, bronze_dir, run_date)

    logger.info("Bronze pipeline complete for date=%s", run_date)
    return df_bronze
