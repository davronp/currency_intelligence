"""run_pipeline.py

Main pipeline orchestrator for the Currency Intelligence Platform.

Execution order
---------------
1. Ingestion   - fetch live exchange rates and save raw JSON
2. Bronze      - convert raw JSON → structured Parquet
3. Silver      - clean, normalise, compute daily returns
4. Gold        - add rolling averages, volatility, z-scores
5. ML          - train Prophet forecasts per currency pair
6. Warehouse   - load gold + forecasts into DuckDB

Usage
-----
    python run_pipeline.py                   # run all stages for today
    python run_pipeline.py --date 2024-01-15 # run for a specific date
    python run_pipeline.py --skip-ml         # skip Prophet training
    python run_pipeline.py --stages bronze silver gold  # selective run
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Make src importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def stage_ingestion(cfg, run_date: date) -> None:
    from src.ingestion.fetch_rates import run_ingestion

    for _target in cfg.currencies.targets:
        logger.info("Ingesting rates for base=USD, target group")
        break  # The free API returns all rates in one call

    run_ingestion(
        base_currency=cfg.currencies.base,
        target_currencies=cfg.currencies.targets,
        raw_dir=cfg.paths.raw,
        base_url=cfg.api.base_url,
        timeout=cfg.api.timeout_seconds,
        retry_attempts=cfg.api.retry_attempts,
        retry_backoff=cfg.api.retry_backoff_seconds,
        run_date=run_date,
    )
    logger.info("✓ Ingestion complete")


def stage_bronze(cfg, spark, run_date: date) -> None:
    from src.bronze.raw_to_bronze import run_bronze

    run_bronze(
        spark=spark,
        raw_dir=cfg.paths.raw,
        bronze_dir=cfg.paths.bronze,
        run_date=run_date,
    )
    logger.info("✓ Bronze layer complete")


def stage_silver(cfg, spark) -> None:
    from src.silver.bronze_to_silver import run_silver

    run_silver(
        spark=spark,
        bronze_dir=cfg.paths.bronze,
        silver_dir=cfg.paths.silver,
        min_rate=cfg.silver.min_valid_rate,
        max_rate=cfg.silver.max_valid_rate,
    )
    logger.info("✓ Silver layer complete")


def stage_gold(cfg, spark) -> None:
    from src.gold.silver_to_gold import run_gold

    run_gold(
        spark=spark,
        silver_dir=cfg.paths.silver,
        gold_dir=cfg.paths.gold,
        rolling_windows=cfg.gold.rolling_windows,
        volatility_window=cfg.gold.volatility_window,
    )
    logger.info("✓ Gold layer complete")


def stage_ml(cfg) -> None:
    from src.ml.forecast import run_forecasting

    prophet_kwargs = {
        "yearly_seasonality": cfg.ml.prophet.yearly_seasonality,
        "weekly_seasonality": cfg.ml.prophet.weekly_seasonality,
        "daily_seasonality": cfg.ml.prophet.daily_seasonality,
        "changepoint_prior_scale": cfg.ml.prophet.changepoint_prior_scale,
        "seasonality_prior_scale": cfg.ml.prophet.seasonality_prior_scale,
    }

    run_forecasting(
        gold_dir=cfg.paths.gold,
        forecasts_dir=cfg.paths.forecasts,
        horizon_days=cfg.ml.forecast_horizon_days,
        min_training_rows=cfg.ml.min_training_rows,
        prophet_kwargs=prophet_kwargs,
        interval_width=cfg.ml.prophet.interval_width,
    )
    logger.info("✓ ML forecasting complete")


def stage_warehouse(cfg) -> None:
    from src.warehouse.load_duckdb import run_warehouse

    run_warehouse(
        db_path=cfg.warehouse.db_file,
        gold_dir=cfg.paths.gold,
        forecasts_dir=cfg.paths.forecasts,
        table_names=cfg.warehouse.tables,
    )
    logger.info("✓ Warehouse load complete")


ALL_STAGES = ["ingestion", "bronze", "silver", "gold", "ml", "warehouse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Currency Intelligence Platform - Pipeline Runner")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Run date in YYYY-MM-DD format (default: today UTC)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=None,
        help="Stages to run (default: all stages)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip the ML forecasting stage",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = load_config()

    # Initialise logging to both console and file
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    log_file = cfg.paths.logs / f"pipeline_{datetime.utcnow():%Y%m%d_%H%M%S}.log"
    setup_logging(level=args.log_level, log_file=log_file)

    run_date = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()

    stages = args.stages or ALL_STAGES
    if args.skip_ml and "ml" in stages:
        stages = [s for s in stages if s != "ml"]

    logger.info("=" * 60)
    logger.info("Currency Intelligence Pipeline")
    logger.info("Run date : %s", run_date)
    logger.info("Stages   : %s", stages)
    logger.info("=" * 60)

    spark = None

    try:
        # Initialise Spark once for all Spark stages
        if any(s in stages for s in ["bronze", "silver", "gold"]):
            from src.utils.spark_utils import get_spark_session

            spark = get_spark_session(
                app_name=cfg.spark.app_name,
                master=cfg.spark.master,
                shuffle_partitions=cfg.spark.shuffle_partitions,
                log_level=cfg.spark.log_level,
                extra_configs=cfg.spark.configs,
            )

        if "ingestion" in stages:
            logger.info("── Stage: ingestion ──")
            stage_ingestion(cfg, run_date)

        if "bronze" in stages:
            logger.info("── Stage: bronze ──")
            stage_bronze(cfg, spark, run_date)

        if "silver" in stages:
            logger.info("── Stage: silver ──")
            stage_silver(cfg, spark)

        if "gold" in stages:
            logger.info("── Stage: gold ──")
            stage_gold(cfg, spark)

        if "ml" in stages:
            logger.info("── Stage: ml ──")
            stage_ml(cfg)

        if "warehouse" in stages:
            logger.info("── Stage: warehouse ──")
            stage_warehouse(cfg)

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully  ✓")
        logger.info("=" * 60)
        return 0

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    finally:
        if spark is not None:
            spark.stop()
            logger.info("SparkSession stopped")


if __name__ == "__main__":
    sys.exit(main())
