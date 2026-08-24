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

from src.ml.metrics import coverage, mae, mape
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = get_logger(__name__)

# Prophet import is deferred so the rest of the codebase can be
# imported without Prophet installed (e.g. in CI without ML deps).
try:
    from prophet import Prophet  # type: ignore[import-untyped]

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
    *,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    interval_width: float = 0.95,
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
        interval_width=interval_width,
    )
    model.fit(df_prophet)
    return model


def generate_forecast(
    model: Prophet,
    horizon_days: int = 30,
) -> pd.DataFrame:
    """Generate *horizon_days* future predictions.

    Parameters
    ----------
    model:
        Fitted Prophet model.
    horizon_days:
        Number of calendar days to forecast.

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
    _interval_width: float = 0.95,
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
    prophet_kwargs = {**(prophet_kwargs or {}), "interval_width": _interval_width}
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
        forecast = generate_forecast(model, horizon_days)
    except Exception:
        logger.exception("Forecast failed for %s", currency_pair)
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


def _make_prophet_forecaster(
    prophet_kwargs: dict | None,
    interval_width: float,
) -> Callable[[pd.DataFrame, pd.Series], pd.DataFrame]:
    """Build a forecaster that fits Prophet on train data and predicts given dates.

    The returned callable maps ``(train_df[ds, y], holdout_ds)`` to a
    DataFrame with ``ds, yhat, yhat_lower, yhat_upper``. Keeping this
    separate lets :func:`backtest_pair` be unit-tested with a stub
    forecaster that never touches Prophet.
    """
    kwargs = {**(prophet_kwargs or {}), "interval_width": interval_width}

    def _forecast(train_df: pd.DataFrame, holdout_ds: pd.Series) -> pd.DataFrame:
        model = train_prophet(train_df, **kwargs)
        future = pd.DataFrame({"ds": pd.to_datetime(list(holdout_ds))})
        predicted = model.predict(future)
        return predicted[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return _forecast


def backtest_pair(
    pair_df: pd.DataFrame,
    currency_pair: str,
    forecaster: Callable[[pd.DataFrame, pd.Series], pd.DataFrame],
    backtest_horizon: int = 30,
    min_training_rows: int = 30,
) -> dict | None:
    """Score forecast accuracy on a held-out tail of a pair's history.

    Trains (via *forecaster*) on all but the last *backtest_horizon*
    observations, predicts those dates, and scores MAE, MAPE and interval
    coverage against the actuals. Returns ``None`` when there is not enough
    history to both train and hold out.

    Parameters
    ----------
    pair_df:
        Gold-layer Pandas DataFrame for one ``currency_pair``.
    currency_pair:
        Label for logging and output.
    forecaster:
        Callable mapping ``(train_df, holdout_ds)`` to a prediction frame.
    backtest_horizon:
        Number of trailing observations to hold out for evaluation.
    min_training_rows:
        Minimum rows required in the training split.

    Returns
    -------
    dict or None
        One row of backtest metrics, or ``None`` if skipped.

    """
    df_prophet = _prepare_prophet_df(pair_df)
    n = len(df_prophet)

    if n < min_training_rows + backtest_horizon:
        logger.warning(
            "Skipping backtest for %s - %d rows (need >= %d)",
            currency_pair,
            n,
            min_training_rows + backtest_horizon,
        )
        return None

    train = df_prophet.iloc[:-backtest_horizon]
    holdout = df_prophet.iloc[-backtest_horizon:]

    predicted = forecaster(train, holdout["ds"])
    merged = holdout.merge(predicted, on="ds", how="inner")
    if merged.empty:
        logger.warning("Backtest for %s produced no aligned predictions", currency_pair)
        return None

    return {
        "currency_pair": currency_pair,
        "backtest_horizon": backtest_horizon,
        "n_train": len(train),
        "n_eval": len(merged),
        "mae": mae(merged["y"], merged["yhat"]),
        "mape": mape(merged["y"], merged["yhat"]),
        "coverage": coverage(merged["y"], merged["yhat_lower"], merged["yhat_upper"]),
        "holdout_start": holdout["ds"].min().date(),
        "holdout_end": holdout["ds"].max().date(),
        "evaluated_at": datetime.now(UTC),
    }


def run_forecasting(
    gold_dir: Path,
    forecasts_dir: Path,
    horizon_days: int = 30,
    backtest_horizon_days: int = 30,
    min_training_rows: int = 30,
    prophet_kwargs: dict | None = None,
    interval_width: float = 0.95,
) -> pd.DataFrame:
    """Full ML pipeline: read gold Parquet -> train per pair -> save forecasts.

    Also runs a walk-forward backtest per pair (train on all but the last
    ``backtest_horizon_days`` observations, score MAE/MAPE/coverage on that
    holdout) and writes the results to ``forecast_metrics.parquet``.

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
    backtest_horizon_days:
        Trailing observations held out for backtest scoring.
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

    forecaster = _make_prophet_forecaster(prophet_kwargs, interval_width)
    all_forecasts: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []

    for pair in pairs:
        pair_df = df_gold[df_gold["currency_pair"] == pair].copy()
        forecast_df = forecast_pair(
            pair_df,
            currency_pair=pair,
            horizon_days=horizon_days,
            min_training_rows=min_training_rows,
            prophet_kwargs=prophet_kwargs,
            _interval_width=interval_width,
        )
        if forecast_df is not None:
            all_forecasts.append(forecast_df)

        try:
            metrics = backtest_pair(
                pair_df,
                currency_pair=pair,
                forecaster=forecaster,
                backtest_horizon=backtest_horizon_days,
                min_training_rows=min_training_rows,
            )
        except Exception:
            logger.exception("Backtest failed for %s", pair)
            metrics = None

        if metrics is not None:
            metrics_rows.append(metrics)
            logger.info(
                "Backtest %s: MAPE=%.3f%% coverage=%.1f%% (n_eval=%d)",
                pair,
                metrics["mape"],
                metrics["coverage"],
                metrics["n_eval"],
            )

    forecasts_dir.mkdir(parents=True, exist_ok=True)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_parquet(forecasts_dir / "forecast_metrics.parquet", index=False)
        logger.info("Backtest metrics saved (%d pairs)", len(metrics_df))

    if not all_forecasts:
        logger.warning("No forecasts were produced")
        return pd.DataFrame()

    combined = pd.concat(all_forecasts, ignore_index=True)
    out_path = forecasts_dir / "forecasts.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Forecasts saved -> %s (%d rows)", out_path, len(combined))

    return combined
