"""src/utils/schema.py.

Explicit PySpark schemas for every medallion layer.

Defining schemas centrally ensures:
  - Bronze, Silver and Gold consumers always agree on column types.
  - Schema drift is caught at write time, not query time.
  - Unit tests can build synthetic DataFrames without Spark inference.
"""

from __future__ import annotations

from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

BRONZE_SCHEMA = StructType(
    [
        StructField("ingestion_date", DateType(), nullable=False),
        StructField("base_currency", StringType(), nullable=False),
        StructField("target_currency", StringType(), nullable=False),
        StructField("rate", DoubleType(), nullable=False),
        StructField("source", StringType(), nullable=False),
        StructField("ingested_at", TimestampType(), nullable=False),
    ]
)


SILVER_SCHEMA = StructType(
    [
        StructField("date", DateType(), nullable=False),
        StructField("currency_pair", StringType(), nullable=False),
        StructField("base_currency", StringType(), nullable=False),
        StructField("target_currency", StringType(), nullable=False),
        StructField("rate", DoubleType(), nullable=False),
        StructField("prev_rate", DoubleType(), nullable=True),
        StructField("daily_return", DoubleType(), nullable=True),
        StructField("processed_at", TimestampType(), nullable=False),
    ]
)


GOLD_SCHEMA = StructType(
    [
        StructField("date", DateType(), nullable=False),
        StructField("currency_pair", StringType(), nullable=False),
        StructField("rate", DoubleType(), nullable=False),
        StructField("daily_return", DoubleType(), nullable=True),
        StructField("ma_7", DoubleType(), nullable=True),
        StructField("ma_30", DoubleType(), nullable=True),
        StructField("ma_90", DoubleType(), nullable=True),
        StructField("volatility_30", DoubleType(), nullable=True),
        StructField("rate_z_score", DoubleType(), nullable=True),
        StructField("created_at", TimestampType(), nullable=False),
    ]
)


FORECAST_SCHEMA = StructType(
    [
        StructField("currency_pair", StringType(), nullable=False),
        StructField("forecast_date", DateType(), nullable=False),
        StructField("yhat", DoubleType(), nullable=False),
        StructField("yhat_lower", DoubleType(), nullable=False),
        StructField("yhat_upper", DoubleType(), nullable=False),
        StructField("model_trained_at", TimestampType(), nullable=False),
        StructField("training_rows", IntegerType(), nullable=False),
    ]
)
