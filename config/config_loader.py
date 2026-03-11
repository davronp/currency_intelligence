"""config/config_loader.py.

Typed configuration loader.  Reads settings.yaml and exposes
strongly-typed dataclasses so every module can import config
objects instead of raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def _project_root() -> Path:
    """Return the project root directory (two levels above this file)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    timezone: str


@dataclass(frozen=True)
class CurrencyConfig:
    base: str
    targets: list[str]

    @property
    def pairs(self) -> list[str]:
        """Return list of 'BASE_TARGET' pair strings."""
        return [f"{self.base}_{t}" for t in self.targets]


@dataclass(frozen=True)
class ApiConfig:
    provider: str
    base_url: str
    timeout_seconds: int
    retry_attempts: int
    retry_backoff_seconds: int


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    forecasts: Path
    warehouse: Path
    logs: Path


@dataclass(frozen=True)
class SparkConfig:
    app_name: str
    master: str
    shuffle_partitions: int
    log_level: str
    configs: dict[str, str]


@dataclass(frozen=True)
class SilverConfig:
    min_valid_rate: float
    max_valid_rate: float


@dataclass(frozen=True)
class GoldConfig:
    rolling_windows: list[int]
    volatility_window: int


@dataclass(frozen=True)
class ProphetConfig:
    yearly_seasonality: bool
    weekly_seasonality: bool
    daily_seasonality: bool
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    interval_width: float


@dataclass(frozen=True)
class MlConfig:
    forecast_horizon_days: int
    prophet: ProphetConfig
    min_training_rows: int


@dataclass(frozen=True)
class WarehouseConfig:
    db_file: Path
    tables: dict[str, str]


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    currencies: CurrencyConfig
    api: ApiConfig
    paths: PathsConfig
    spark: SparkConfig
    silver: SilverConfig
    gold: GoldConfig
    ml: MlConfig
    warehouse: WarehouseConfig


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate the YAML configuration file.

    Parameters
    ----------
    config_path:
        Optional explicit path to settings.yaml.
        Defaults to <project_root>/config/settings.yaml.

    Returns
    -------
    AppConfig
        Fully validated, immutable configuration object.

    """
    root = _project_root()

    if config_path is None:
        config_path = root / "config" / "settings.yaml"

    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    def _abs(rel: str) -> Path:
        """Resolve a relative path string against the project root."""
        p = Path(rel)
        return p if p.is_absolute() else root / p

    paths_raw = raw["paths"]
    paths = PathsConfig(
        raw=_abs(paths_raw["raw"]),
        bronze=_abs(paths_raw["bronze"]),
        silver=_abs(paths_raw["silver"]),
        gold=_abs(paths_raw["gold"]),
        forecasts=_abs(paths_raw["forecasts"]),
        warehouse=_abs(paths_raw["warehouse"]),
        logs=_abs(paths_raw["logs"]),
    )

    prophet_raw = raw["ml"]["prophet"]
    prophet = ProphetConfig(
        yearly_seasonality=prophet_raw["yearly_seasonality"],
        weekly_seasonality=prophet_raw["weekly_seasonality"],
        daily_seasonality=prophet_raw["daily_seasonality"],
        changepoint_prior_scale=prophet_raw["changepoint_prior_scale"],
        seasonality_prior_scale=prophet_raw["seasonality_prior_scale"],
        interval_width=prophet_raw["interval_width"],
    )

    warehouse_raw = raw["warehouse"]
    warehouse = WarehouseConfig(
        db_file=_abs(warehouse_raw["db_file"]),
        tables=warehouse_raw["tables"],
    )

    return AppConfig(
        project=ProjectConfig(**raw["project"]),
        currencies=CurrencyConfig(**raw["currencies"]),
        api=ApiConfig(**raw["api"]),
        paths=paths,
        spark=SparkConfig(**raw["spark"]),
        silver=SilverConfig(**raw["silver"]),
        gold=GoldConfig(**raw["gold"]),
        ml=MlConfig(
            forecast_horizon_days=raw["ml"]["forecast_horizon_days"],
            prophet=prophet,
            min_training_rows=raw["ml"]["min_training_rows"],
        ),
        warehouse=warehouse,
    )
