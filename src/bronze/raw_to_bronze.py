"""src/bronze/raw_to_bronze.py.

Bronze layer: read the raw JSONL file(s), parse into a structured
Spark DataFrame, and write a single Parquet file.

Storage layout
--------------
    data/raw/rates_USD.jsonl   ← input  (one JSON line per date)
    data/bronze/bronze.parquet ← output (single coalesced file)

Idempotency: the output is always fully overwritten.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.logger import get_logger
from src.utils.schema import BRONZE_SCHEMA
from src.utils.spark_utils import deduplicate, enforce_schema

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def _parse_jsonl_file(jsonl_path: Path) -> list[dict]:
    """Parse a ``rates_<BASE>.jsonl`` file into flat row dicts.

    Each line is one date's payload; each currency pair within that
    payload becomes one row.
    """
    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line in %s: %s", jsonl_path.name, exc)
                continue

            base = payload["base"]
            record_date = payload.get("date", "")
            ingested_at = payload.get("fetched_at", datetime.now(UTC).isoformat())
            source = payload.get("source", "unknown")

            for target, rate in payload.get("rates", {}).items():
                rows.append(
                    {
                        "ingestion_date": record_date,
                        "base_currency": base,
                        "target_currency": target,
                        "rate": float(rate),
                        "source": source,
                        "ingested_at": ingested_at,
                    }
                )

    logger.debug("Parsed %d rows from %s", len(rows), jsonl_path.name)
    return rows


def load_raw_files(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Load all ``rates_*.jsonl`` files found in *raw_dir*.

    Parameters
    ----------
    spark:
        Active SparkSession.
    raw_dir:
        Root of the raw data lake (``data/raw``).

    Returns
    -------
    DataFrame
        Unvalidated bronze-shaped DataFrame.

    """
    jsonl_files = sorted(raw_dir.glob("rates_*.jsonl"))
    if not jsonl_files:
        msg = f"No rates_*.jsonl files found in: {raw_dir}"
        raise FileNotFoundError(msg)

    logger.info("Loading %d JSONL file(s) from %s", len(jsonl_files), raw_dir)

    all_rows: list[dict] = []
    for jf in jsonl_files:
        all_rows.extend(_parse_jsonl_file(jf))

    if not all_rows:
        msg = f"No rows parsed from JSONL files in {raw_dir}"
        raise ValueError(msg)

    df = spark.createDataFrame(all_rows)
    logger.info("Loaded %d raw row(s) from JSONL", df.count())
    return df


def transform_to_bronze(df: DataFrame) -> DataFrame:
    """Cast columns, enforce schema, and deduplicate.

    Deduplication key: (ingestion_date, base_currency, target_currency),
    keeping the row with the latest ingested_at.
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


def write_bronze(df: DataFrame, bronze_dir: Path) -> None:
    """Write the full bronze dataset as a single Parquet file.

    Using one file avoids the per-date partition explosion while still
    giving DuckDB a clean Parquet surface to query.
    """
    bronze_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(bronze_dir / "bronze.parquet")
    df.coalesce(1).write.mode("overwrite").format("parquet").save(out_path)
    logger.info("Bronze layer written -> %s", out_path)


def run_bronze(
    spark: SparkSession,
    raw_dir: Path,
    bronze_dir: Path,
    _run_date: date | None = None,  # kept for API compatibility, no longer used
) -> DataFrame:
    """End-to-end bronze pipeline: load all JSONL -> transform -> write.

    Returns the bronze DataFrame for downstream chaining.
    """
    logger.info("Starting bronze pipeline")
    df_raw = load_raw_files(spark, raw_dir)
    df_bronze = transform_to_bronze(df_raw)
    write_bronze(df_bronze, bronze_dir)
    logger.info("Bronze pipeline complete")
    return df_bronze
