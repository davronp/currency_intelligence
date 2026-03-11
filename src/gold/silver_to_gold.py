"""src/gold/silver_to_gold.py.

Gold layer: generate analytical features on top of the clean
silver data and write the curated dataset to Parquet.

Features produced
-----------------
- Moving averages  : ma_7, ma_30, ma_90
- Volatility       : volatility_30  (rolling 30-day std-dev of returns)
- Z-score          : rate_z_score   (lifetime z-score per currency pair)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.logger import get_logger
from src.utils.schema import GOLD_SCHEMA
from src.utils.spark_utils import (
    deduplicate,
    enforce_schema,
    read_parquet,
    rolling_average,
    rolling_stddev,
    write_parquet,
    z_score,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def add_moving_averages(
    df: DataFrame,
    windows: list[int],
    partition_cols: list[str] = ("currency_pair",),
    order_col: str = "date",
    value_col: str = "rate",
) -> DataFrame:
    """Add moving-average columns for each window size.

    Parameters
    ----------
    df:
        Silver DataFrame.
    windows:
        List of window sizes in days, e.g. ``[7, 30, 90]``.
    partition_cols:
        Window partition columns.
    order_col:
        Sort column for the window.
    value_col:
        Column to average.

    Returns
    -------
    DataFrame with ``ma_<N>`` columns appended.

    """
    for w in windows:
        df = rolling_average(
            df,
            partition_cols=list(partition_cols),
            order_col=order_col,
            value_col=value_col,
            window_days=w,
            alias=f"ma_{w}",
        )
    return df


def add_volatility(
    df: DataFrame,
    window: int = 30,
    partition_cols: list[str] = ("currency_pair",),
    order_col: str = "date",
    return_col: str = "daily_return",
) -> DataFrame:
    """Add a rolling volatility column (std-dev of daily returns).

    Parameters
    ----------
    df:
        Silver DataFrame with a ``daily_return`` column.
    window:
        Look-back window in days.

    Returns
    -------
    DataFrame with ``volatility_<window>`` appended.

    """
    return rolling_stddev(
        df,
        partition_cols=list(partition_cols),
        order_col=order_col,
        value_col=return_col,
        window_days=window,
        alias=f"volatility_{window}",
    )


def add_rate_z_score(
    df: DataFrame,
    value_col: str = "rate",
    partition_cols: list[str] = ("currency_pair",),
) -> DataFrame:
    """Append a lifetime z-score for the rate within each currency pair.

    A z-score > 2 or < -2 flags a statistically unusual rate.
    """
    return z_score(df, value_col, list(partition_cols), alias="rate_z_score")


def add_created_timestamp(df: DataFrame) -> DataFrame:
    """Append a UTC creation timestamp."""
    return df.withColumn("created_at", F.current_timestamp())


def transform_to_gold(
    df: DataFrame,
    rolling_windows: list[int] = (7, 30, 90),
    volatility_window: int = 30,
) -> DataFrame:
    """Full gold transformation pipeline.

    Steps
    -----
    1. Add moving averages.
    2. Add volatility.
    3. Add z-score.
    4. Add created_at timestamp.
    5. Enforce gold schema.
    6. Deduplicate on (date, currency_pair).

    Parameters
    ----------
    df:
        Silver DataFrame.
    rolling_windows:
        Window sizes for moving averages.
    volatility_window:
        Window size for volatility calculation.

    Returns
    -------
    DataFrame
        Enriched gold DataFrame.

    """
    df = add_moving_averages(df, list(rolling_windows))
    df = add_volatility(df, volatility_window)
    df = add_rate_z_score(df)
    df = add_created_timestamp(df)
    df = enforce_schema(df, GOLD_SCHEMA)
    df = deduplicate(df, ["date", "currency_pair"], "created_at")
    logger.info("Gold transformation complete (%d rows)", df.count())
    return df


def write_gold(df: DataFrame, gold_dir: Path) -> None:
    """Write the gold dataset as a single Parquet file.

    Coalescing to 1 keeps the lake lean — the full gold dataset for
    years of daily data for a handful of pairs is still well under 50 MB.
    """
    write_parquet(df.coalesce(1), str(gold_dir), partition_by=None, mode="overwrite")
    logger.info("Gold layer written → %s", gold_dir)


def run_gold(
    spark: SparkSession,
    silver_dir: Path,
    gold_dir: Path,
    rolling_windows: list[int] = (7, 30, 90),
    volatility_window: int = 30,
) -> DataFrame:
    """End-to-end gold pipeline: read silver → transform → write.

    Returns the gold DataFrame for downstream chaining.
    """
    logger.info("Starting gold pipeline")
    df_silver = read_parquet(spark, str(silver_dir))
    df_gold = transform_to_gold(df_silver, rolling_windows, volatility_window)
    write_gold(df_gold, gold_dir)
    logger.info("Gold pipeline complete")
    return df_gold
