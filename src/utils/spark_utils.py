"""src/utils/spark_utils.py.

Spark session factory and reusable transformation helpers.

All Spark helper functions are pure: they accept a DataFrame and
return a new DataFrame so they can be composed and unit-tested
in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, TimestampType

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from pyspark.sql.types import StructType

logger = get_logger(__name__)


def get_spark_session(
    app_name: str = "CurrencyIntelligence",
    master: str = "local[*]",
    shuffle_partitions: int = 4,
    log_level: str = "WARN",
    extra_configs: dict | None = None,
) -> SparkSession:
    """Build or retrieve an existing SparkSession.

    Parameters
    ----------
    app_name:
        Name visible in the Spark UI.
    master:
        Spark master URL (``local[*]`` for local mode).
    shuffle_partitions:
        ``spark.sql.shuffle.partitions`` - keep low for local dev.
    log_level:
        Spark log level string.
    extra_configs:
        Any additional ``spark.`` config key/value pairs.

    Returns
    -------
    SparkSession

    """
    logger.info("Initialising SparkSession: app=%s master=%s", app_name, master)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )

    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, str(value))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    logger.info("SparkSession ready (version=%s)", spark.version)
    return spark


def add_processing_timestamp(df: DataFrame, col_name: str = "processed_at") -> DataFrame:
    """Append a UTC timestamp column."""
    return df.withColumn(col_name, F.current_timestamp())


def enforce_schema(df: DataFrame, schema: StructType) -> DataFrame:
    """Select and cast columns to match *schema* exactly.

    Extra columns in *df* are dropped; missing columns raise ValueError.
    """
    expected = {f.name: f.dataType for f in schema.fields}
    missing = set(expected) - set(df.columns)
    if missing:
        msg = f"DataFrame is missing required columns: {missing}"
        raise ValueError(msg)

    return df.select(*[F.col(name).cast(dtype).alias(name) for name, dtype in expected.items()])


def deduplicate(df: DataFrame, partition_cols: list[str], order_col: str) -> DataFrame:
    """Keep the latest row per *partition_cols*, ordered descending by *order_col*.
    Ensures idempotency when a pipeline re-runs.
    """
    window = Window.partitionBy(*partition_cols).orderBy(F.col(order_col).desc())
    return df.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")


def _orderable(df: DataFrame, col_name: str) -> F.Column:
    """Return an orderable (numeric) expression for *col_name*.

    Spark 4.x no longer allows ``CAST(DATE AS BIGINT)`` in window
    ORDER BY clauses.  Use ``unix_date()`` for DateType columns and
    ``unix_timestamp()`` for TimestampType so the window always
    orders over a plain integer, which is safe across all Spark versions.
    """
    col_type = df.schema[col_name].dataType
    if isinstance(col_type, DateType):
        return F.unix_date(F.col(col_name))
    if isinstance(col_type, TimestampType):
        return F.unix_timestamp(F.col(col_name))
    # Already numeric - use as-is
    return F.col(col_name)


def rolling_average(
    df: DataFrame,
    partition_cols: list[str],
    order_col: str,
    value_col: str,
    window_days: int,
    alias: str | None = None,
) -> DataFrame:
    """Add a rolling (moving) average column using a row-based window.

    Parameters
    ----------
    df:
        Input DataFrame.
    partition_cols:
        Columns to partition the window by (e.g. ``["currency_pair"]``).
    order_col:
        Date, timestamp, or numeric column to order within each partition.
    value_col:
        Column to average.
    window_days:
        Look-back window size in rows.
    alias:
        Output column name (default: ``ma_{window_days}``).

    Returns
    -------
    DataFrame with an additional moving-average column.

    """
    alias = alias or f"ma_{window_days}"
    window = (
        Window.partitionBy(*partition_cols)
        .orderBy(_orderable(df, order_col))
        .rowsBetween(-(window_days - 1), 0)
    )
    return df.withColumn(alias, F.avg(F.col(value_col)).over(window))


def rolling_stddev(
    df: DataFrame,
    partition_cols: list[str],
    order_col: str,
    value_col: str,
    window_days: int,
    alias: str | None = None,
) -> DataFrame:
    """Add a rolling standard-deviation (volatility) column."""
    alias = alias or f"stddev_{window_days}"
    window = (
        Window.partitionBy(*partition_cols)
        .orderBy(_orderable(df, order_col))
        .rowsBetween(-(window_days - 1), 0)
    )
    return df.withColumn(alias, F.stddev(F.col(value_col)).over(window))


def z_score(
    df: DataFrame,
    value_col: str,
    partition_cols: list[str],
    alias: str = "z_score",
) -> DataFrame:
    """Append a z-score column (value - mean) / stddev
    computed within *partition_cols*.
    """
    window = Window.partitionBy(*partition_cols)
    return df.withColumn(
        alias,
        (F.col(value_col) - F.avg(F.col(value_col)).over(window)) / F.stddev(F.col(value_col)).over(window),
    )


def write_parquet(
    df: DataFrame,
    path: str,
    partition_by: list[str] | None = None,
    mode: str = "overwrite",
) -> None:
    """Write a DataFrame to Parquet with optional partitioning.

    Parameters
    ----------
    df:
        DataFrame to persist.
    path:
        Output directory path.
    partition_by:
        Column names to partition the output by.
    mode:
        Spark write mode.  Defaults to ``"overwrite"`` (idempotent).

    """
    writer = df.write.mode(mode).format("parquet")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(path)
    logger.info("Wrote Parquet -> %s (mode=%s)", path, mode)


def read_parquet(spark: SparkSession, path: str, schema: StructType | None = None) -> DataFrame:
    """Read a Parquet dataset, optionally enforcing a schema."""
    reader = spark.read.format("parquet")
    if schema:
        reader = reader.schema(schema)
    df = reader.load(path)
    logger.info("Read Parquet ← %s (%d rows)", path, df.count())
    return df
