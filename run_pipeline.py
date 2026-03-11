"""run_pipeline.py.

Main pipeline orchestrator for the Currency Intelligence Platform.

Execution order
---------------
1. Ingestion   - fetch live/historical exchange rates and save raw JSON
2. Bronze      - convert raw JSON -> structured Parquet
3. Silver      - clean, normalise, compute daily returns
4. Gold        - add rolling averages, volatility, z-scores
5. ML          - train Prophet forecasts per currency pair
6. Warehouse   - load gold + forecasts into DuckDB

Usage
-----
    # Run all stages for today only
    python run_pipeline.py

    # Backfill the last 90 days (fetches history + rebuilds all layers)
    python run_pipeline.py --backfill-days 90

    # Run for an explicit date range
    python run_pipeline.py --start-date 2025-01-01 --end-date 2025-03-11

    # Run for a single specific date
    python run_pipeline.py --date 2025-06-01

    # Skip ML forecasting
    python run_pipeline.py --backfill-days 90 --skip-ml

    # Run only specific stages (e.g. re-build gold after a config change)
    python run_pipeline.py --stages silver gold warehouse
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config
from src.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def stage_ingestion(cfg, run_date: date, force: bool = False) -> None:
    from src.ingestion.fetch_rates import run_ingestion

    run_ingestion(
        base_currency=cfg.currencies.base,
        target_currencies=cfg.currencies.targets,
        raw_dir=cfg.paths.raw,
        base_url=cfg.api.base_url,
        run_date=run_date,
        timeout=cfg.api.timeout_seconds,
        retry_attempts=cfg.api.retry_attempts,
        retry_backoff=cfg.api.retry_backoff_seconds,
        force=force,
    )


def stage_bronze(cfg, spark, run_date: date) -> None:
    from src.bronze.raw_to_bronze import run_bronze

    run_bronze(
        spark=spark,
        raw_dir=cfg.paths.raw,
        bronze_dir=cfg.paths.bronze,
        run_date=run_date,
    )


def stage_silver(cfg, spark) -> None:
    from src.silver.bronze_to_silver import run_silver

    run_silver(
        spark=spark,
        bronze_dir=cfg.paths.bronze,
        silver_dir=cfg.paths.silver,
        min_rate=cfg.silver.min_valid_rate,
        max_rate=cfg.silver.max_valid_rate,
    )


def stage_gold(cfg, spark) -> None:
    from src.gold.silver_to_gold import run_gold

    run_gold(
        spark=spark,
        silver_dir=cfg.paths.silver,
        gold_dir=cfg.paths.gold,
        rolling_windows=cfg.gold.rolling_windows,
        volatility_window=cfg.gold.volatility_window,
    )


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


def stage_warehouse(cfg) -> None:
    from src.warehouse.load_duckdb import run_warehouse

    run_warehouse(
        db_path=cfg.warehouse.db_file,
        gold_dir=cfg.paths.gold,
        forecasts_dir=cfg.paths.forecasts,
        table_names=cfg.warehouse.tables,
    )


def _date_range(start: date, end: date) -> list[date]:
    """Return every calendar date from *start* to *end* inclusive."""
    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


def _skip_weekends(dates: list[date]) -> list[date]:
    """Filter out Saturdays and Sundays.

    Frankfurter returns weekday-only ECB rates; weekend dates return the
    most recent Friday's rates (same date string), which would create
    duplicate rows.
    """
    return [d for d in dates if d.weekday() < 5]


ALL_STAGES = ["ingestion", "bronze", "silver", "gold", "ml", "warehouse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Currency Intelligence Platform - Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date",
        type=str,
        default=None,
        help="Single run date YYYY-MM-DD (default: today)",
    )
    date_group.add_argument(
        "--backfill-days",
        type=int,
        default=None,
        metavar="N",
        help="Fetch the last N calendar days of history (e.g. --backfill-days 90)",
    )
    date_group.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start of an explicit date range (pair with --end-date)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End of the date range (default: today, used with --start-date)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=None,
        help="Stages to run (default: all)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip ML forecasting stage",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="Re-fetch raw files even if they already exist on disk",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def resolve_dates(args: argparse.Namespace) -> list[date]:
    """Return the ordered list of dates to ingest."""
    today = date.today()

    if args.backfill_days:
        start = today - timedelta(days=args.backfill_days - 1)
        dates = _date_range(start, today)
    elif args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else today
        dates = _date_range(start, end)
    elif args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    else:
        dates = [today]

    # Frankfurter only has weekday data — skip weekends to avoid dupes
    weekday_dates = _skip_weekends(dates)
    skipped = len(dates) - len(weekday_dates)
    if skipped:
        logger.info("Skipping %d weekend date(s) (Frankfurter weekdays only)", skipped)

    return weekday_dates


def main() -> int:
    args = parse_args()
    cfg = load_config()

    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    log_file = cfg.paths.logs / "pipeline.log"  # single rotating file, not one per run
    setup_logging(level=args.log_level, log_file=log_file)

    stages = args.stages or ALL_STAGES
    if args.skip_ml and "ml" in stages:
        stages = [s for s in stages if s != "ml"]

    dates = resolve_dates(args)
    n = len(dates)

    logger.info("=" * 60)
    logger.info("Currency Intelligence Pipeline")
    logger.info("Dates    : %s → %s (%d day(s))", dates[0], dates[-1], n)
    logger.info("Stages   : %s", stages)
    logger.info("=" * 60)

    spark = None

    try:
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
            logger.info("── Stage: ingestion (%d dates) ──", n)
            for i, d in enumerate(dates, 1):
                logger.info("  [%d/%d] ingesting %s", i, n, d)
                try:
                    stage_ingestion(cfg, d, force=args.force_ingest)
                except Exception as exc:
                    # Log and continue — a missing holiday/weekend date
                    # from the API should not abort the whole backfill
                    logger.warning("  Ingestion failed for %s: %s (skipping)", d, exc)
            logger.info("✓ Ingestion complete")

        if "bronze" in stages:
            logger.info("── Stage: bronze (%d dates) ──", n)
            for i, d in enumerate(dates, 1):
                raw_dir = cfg.paths.raw / d.isoformat()
                if not raw_dir.exists():
                    logger.warning("  No raw data for %s, skipping bronze", d)
                    continue
                logger.info("  [%d/%d] bronze %s", i, n, d)
                try:
                    stage_bronze(cfg, spark, d)
                except Exception as exc:
                    logger.warning("  Bronze failed for %s: %s (skipping)", d, exc)
            logger.info("✓ Bronze complete")

        if "silver" in stages:
            logger.info("── Stage: silver ──")
            stage_silver(cfg, spark)
            logger.info("✓ Silver complete")

        if "gold" in stages:
            logger.info("── Stage: gold ──")
            stage_gold(cfg, spark)
            logger.info("✓ Gold complete")

        if "ml" in stages:
            logger.info("── Stage: ml ──")
            stage_ml(cfg)
            logger.info("✓ ML complete")

        if "warehouse" in stages:
            logger.info("── Stage: warehouse ──")
            stage_warehouse(cfg)
            logger.info("✓ Warehouse complete")

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
