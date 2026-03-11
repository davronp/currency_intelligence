"""src/ml/forecast.py.

ML pipeline: train a Facebook Prophet model per currency pair
and persist predictions as Parquet.

Design notes
------------
- Models are trained on the full gold-layer history.
- Forecasts are produced for ``forecast_horizon_days`` into the future.
- Predictions are stored with uncertainty intervals (yhat_lower / yhat_upper).
- Training is skipped if fewer than ``min_training_rows`` observations exist.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

# Prophet import is deferred so the rest of the codebase can be
# imported without Prophet installed (e.g. in CI without ML deps).
try:
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed - ML pipeline will be unavailable")


def _validate_prophet_available() -> None:
    if not PROPHET_AVAILABLE:
        msg = "prophet is not installed.  Run: pip install prophet"
        raise ImportError(msg)


def _prepare_prophet_df(df_pair: pd.DataFrame) -> pd.DataFrame:
    """Convert a gold-layer pair DataFrame to Prophet's ``ds``/``y`` format.

    Parameters
    ----------
    df_pair:
        Pandas DataFrame filtered to a single currency pair,
        with columns ``date`` and ``rate``.

    Returns
    -------
    pd.DataFrame with columns ``ds`` (datetime) and ``y`` (float).

    """
    df_prophet = df_pair[["date", "rate"]].rename(columns={"date": "ds", "rate": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
    return df_prophet.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)


def train_prophet(
    df_prophet: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
) -> Prophet:
    """Fit a Prophet model on the prepared DataFrame.

    Parameters
    ----------
    df_prophet:
        DataFrame with ``ds`` and ``y`` columns.
    All other parameters map directly to Prophet constructor kwargs.

    Returns
    -------
    Fitted Prophet model.

    """
    _validate_prophet_available()
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    model.fit(df_prophet)
    return model


def generate_forecast(
    model: Prophet,
    horizon_days: int = 30,
    interval_width: float = 0.95,
) -> pd.DataFrame:
    """Generate *horizon_days* future predictions.

    Parameters
    ----------
    model:
        Fitted Prophet model.
    horizon_days:
        Number of calendar days to forecast.
    interval_width:
        Probability mass of the uncertainty interval.

    Returns
    -------
    pd.DataFrame with columns ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.

    """
    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)


def forecast_pair(
    pair_df: pd.DataFrame,
    currency_pair: str,
    horizon_days: int = 30,
    min_training_rows: int = 30,
    prophet_kwargs: dict | None = None,
    interval_width: float = 0.95,
) -> pd.DataFrame | None:
    """Train Prophet and produce forecasts for a single currency pair.

    Parameters
    ----------
    pair_df:
        Gold-layer Pandas DataFrame for one ``currency_pair``.
    currency_pair:
        Label for logging and output.
    horizon_days:
        Forecast horizon.
    min_training_rows:
        Minimum rows required to proceed.
    prophet_kwargs:
        Extra kwargs forwarded to :func:`train_prophet`.
    interval_width:
        Forecast uncertainty interval width.

    Returns
    -------
    pd.DataFrame or None
        Flat forecast rows, or ``None`` if skipped.

    """
    prophet_kwargs = prophet_kwargs or {}
    n_rows = len(pair_df)

    if n_rows < min_training_rows:
        logger.warning(
            "Skipping %s - only %d rows (min=%d)",
            currency_pair,
            n_rows,
            min_training_rows,
        )
        return None

    logger.info("Training Prophet for %s (%d rows)", currency_pair, n_rows)
    df_prophet = _prepare_prophet_df(pair_df)

    try:
        model = train_prophet(df_prophet, **prophet_kwargs)
        forecast = generate_forecast(model, horizon_days, interval_width)
    except Exception as exc:
        logger.exception("Forecast failed for %s: %s", currency_pair, exc)
        return None

    trained_at = datetime.now(UTC)
    forecast = forecast.assign(
        currency_pair=currency_pair,
        forecast_date=forecast["ds"].dt.date,
        model_trained_at=trained_at,
        training_rows=n_rows,
    ).drop(columns=["ds"])

    logger.info("Forecast complete for %s - %d future points", currency_pair, len(forecast))
    return forecast


def run_forecasting(
    gold_dir: Path,
    forecasts_dir: Path,
    horizon_days: int = 30,
    min_training_rows: int = 30,
    prophet_kwargs: dict | None = None,
    interval_width: float = 0.95,
) -> pd.DataFrame:
    """Full ML pipeline: read gold Parquet → train per pair → save forecasts.

    Uses Pandas directly (no Spark) because Prophet is a single-machine
    library and the gold dataset fits comfortably in memory.

    Parameters
    ----------
    gold_dir:
        Path to the gold Parquet lake.
    forecasts_dir:
        Output directory for forecast Parquet files.
    horizon_days:
        Forecast horizon in calendar days.
    min_training_rows:
        Skip training if fewer rows exist.
    prophet_kwargs:
        Extra kwargs forwarded to :func:`train_prophet`.
    interval_width:
        Confidence interval width.

    Returns
    -------
    pd.DataFrame
        Combined forecasts for all pairs.

    """
    _validate_prophet_available()
    logger.info("Starting ML forecasting pipeline")

    df_gold = pd.read_parquet(gold_dir)

    if "date" not in df_gold.columns:
        # Handle Spark-partitioned Parquet (partition column in path)
        df_gold = df_gold.reset_index()

    pairs = df_gold["currency_pair"].unique()
    logger.info("Currency pairs found: %s", list(pairs))

    all_forecasts: list[pd.DataFrame] = []

    for pair in pairs:
        pair_df = df_gold[df_gold["currency_pair"] == pair].copy()
        forecast_df = forecast_pair(
            pair_df,
            currency_pair=pair,
            horizon_days=horizon_days,
            min_training_rows=min_training_rows,
            prophet_kwargs=prophet_kwargs,
            interval_width=interval_width,
        )
        if forecast_df is not None:
            all_forecasts.append(forecast_df)

    if not all_forecasts:
        logger.warning("No forecasts were produced")
        return pd.DataFrame()

    combined = pd.concat(all_forecasts, ignore_index=True)

    forecasts_dir.mkdir(parents=True, exist_ok=True)
    out_path = forecasts_dir / "forecasts.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Forecasts saved → %s (%d rows)", out_path, len(combined))

    return combined
