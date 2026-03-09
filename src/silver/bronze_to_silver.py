"""src/silver/bronze_to_silver.py

Silver layer: validate and clean bronze data, build canonical
currency-pair names, compute daily log-returns, and write Parquet.

All transformations are implemented as pure functions for testability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.utils.logger import get_logger
from src.utils.schema import SILVER_SCHEMA
from src.utils.spark_utils import (
    deduplicate,
    enforce_schema,
    read_parquet,
    write_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def filter_valid_rates(
    df: DataFrame,
    min_rate: float = 1e-6,
    max_rate: float = 1e5,
) -> DataFrame:
    """Remove rows with null or out-of-range exchange rates.

    Parameters
    ----------
    df:
        Bronze DataFrame.
    min_rate / max_rate:
        Inclusive bounds for a valid rate value.

    Returns
    -------
    DataFrame
        Filtered DataFrame.

    """
    n_before = df.count()
    df = df.filter(F.col("rate").isNotNull() & F.col("rate").between(min_rate, max_rate))
    n_after = df.count()
    dropped = n_before - n_after
    if dropped:
        logger.warning("Dropped %d row(s) with invalid rates", dropped)
    return df


def add_currency_pair(df: DataFrame) -> DataFrame:
    """Add a canonical ``currency_pair`` column (e.g. ``USD_EUR``).

    Parameters
    ----------
    df:
        Must contain ``base_currency`` and ``target_currency``.

    Returns
    -------
    DataFrame with a new ``currency_pair`` column.

    """
    return df.withColumn(
        "currency_pair",
        F.concat_ws("_", F.col("base_currency"), F.col("target_currency")),
    )


def add_date_column(df: DataFrame) -> DataFrame:
    """Rename ``ingestion_date`` → ``date`` for semantic clarity."""
    return df.withColumnRenamed("ingestion_date", "date")


def compute_daily_returns(df: DataFrame) -> DataFrame:
    """Compute the previous day's rate and the daily percentage return.

    Uses a ``LAG(1)`` window ordered by ``date`` within each ``currency_pair``.

    ``daily_return = (rate / prev_rate) - 1``

    Returns
    -------
    DataFrame with new columns ``prev_rate`` and ``daily_return``.

    """
    window = Window.partitionBy("currency_pair").orderBy("date")

    df = df.withColumn("prev_rate", F.lag("rate", 1).over(window))
    return df.withColumn(
        "daily_return",
        F.when(
            F.col("prev_rate").isNotNull() & (F.col("prev_rate") != 0),
            (F.col("rate") / F.col("prev_rate")) - 1.0,
        ).otherwise(F.lit(None).cast("double")),
    )


def add_processed_timestamp(df: DataFrame) -> DataFrame:
    """Append a UTC processing timestamp."""
    return df.withColumn("processed_at", F.current_timestamp())


def read_all_bronze(spark: SparkSession, bronze_dir: Path) -> DataFrame:
    """Read the entire bronze Parquet lake (all dates).

    The partition column ``ingestion_date`` is reconstructed from the
    folder name via Spark's automatic partition discovery.
    """
    df = read_parquet(spark, str(bronze_dir))
    logger.info("Read bronze lake: %d rows", df.count())
    return df


def transform_to_silver(
    df: DataFrame,
    min_rate: float = 1e-6,
    max_rate: float = 1e5,
) -> DataFrame:
    """Full silver transformation pipeline.

    Steps
    -----
    1. Filter invalid rates.
    2. Add ``currency_pair`` column.
    3. Rename date column.
    4. Compute daily returns.
    5. Enforce silver schema.
    6. Deduplicate on (date, currency_pair).

    Parameters
    ----------
    df:
        Bronze DataFrame (may span multiple dates).
    min_rate / max_rate:
        Passed through to :func:`filter_valid_rates`.

    Returns
    -------
    DataFrame
        Clean, normalised silver DataFrame.

    """
    df = filter_valid_rates(df, min_rate, max_rate)
    df = add_currency_pair(df)
    df = add_date_column(df)
    df = compute_daily_returns(df)
    df = add_processed_timestamp(df)
    df = enforce_schema(df, SILVER_SCHEMA)
    df = deduplicate(df, ["date", "currency_pair"], "processed_at")
    logger.info("Silver transformation complete (%d rows)", df.count())
    return df


def write_silver(df: DataFrame, silver_dir: Path) -> None:
    """Write the full silver dataset, partitioned by ``date``.

    The entire silver dataset is overwritten to ensure idempotency
    when historical data is reprocessed.
    """
    write_parquet(df, str(silver_dir), partition_by=["date"], mode="overwrite")
    logger.info("Silver layer written → %s", silver_dir)


def run_silver(
    spark: SparkSession,
    bronze_dir: Path,
    silver_dir: Path,
    min_rate: float = 1e-6,
    max_rate: float = 1e5,
) -> DataFrame:
    """End-to-end silver pipeline: read bronze → transform → write.

    Returns the silver DataFrame for downstream chaining.
    """
    logger.info("Starting silver pipeline")
    df_bronze = read_all_bronze(spark, bronze_dir)
    df_silver = transform_to_silver(df_bronze, min_rate, max_rate)
    write_silver(df_silver, silver_dir)
    logger.info("Silver pipeline complete")
    return df_silver
