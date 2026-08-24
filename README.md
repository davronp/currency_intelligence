# Currency Intelligence Platform

An end-to-end data engineering pipeline that ingests live FX rates, processes them through a **Medallion architecture** (Raw -> Bronze -> Silver -> Gold), trains **Prophet time-series forecasts** (with walk-forward backtesting), and serves results through an interactive **Streamlit dashboard** - updated daily via **GitHub Actions CI/CD**.

**Live demo:** [currency-intelligence.streamlit.app](https://currencyintelligence.streamlit.app/)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Python, Requests, Frankfurter API (ECB data) |
| Transformation | PySpark 4.1 |
| Storage | Apache Parquet, DuckDB 1.4 |
| Forecasting | Facebook Prophet |
| Dashboard | Streamlit, Plotly |
| CI/CD | GitHub Actions |
| Packaging | uv (pyproject + uv.lock) |
| Code quality | Ruff, pytest, pytest-cov |

---

## Architecture

```
Frankfurter API
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw       │ -> │   Bronze    │ -> │   Silver    │ -> │    Gold     │
│  JSONL      │    │  Parquet    │    │  Parquet    │    │  Parquet    │
│  (per base) │    │  (parsed)   │    │  (cleaned)  │    │  (features) │
└─────────────┘    └─────────────┘    └─────────────┘    └───────┬─────┘
                                                                 │
                                            ┌────────────────────┤
                                            ▼                    ▼
                                   ┌─────────────────┐  ┌──────────────┐
                                   │  ML Forecasting │  │    DuckDB    │
                                   │  (Prophet)      │  │  Warehouse   │
                                   └────────┬────────┘  └───────┬──────┘
                                            └──────────┬────────┘
                                                       ▼
                                            ┌─────────────────────┐
                                            │  Streamlit Dashboard│
                                            └─────────────────────┘
```

### Medallion Layers

| Layer | Format | What happens |
|---|---|---|
| **Raw** | JSONL | Verbatim API responses, one file per base currency, upserted by date |
| **Bronze** | Parquet | Parsed into rows, schema-enforced, deduplicated |
| **Silver** | Parquet | Rate validation, currency pair labels, daily returns via window functions |
| **Gold** | Parquet | MA-7/30/90, 30-day volatility, z-scores - all computed with Spark window functions |
| **Forecasts** | Parquet | 30-day Prophet predictions with 95% confidence intervals per pair |
| **Warehouse** | DuckDB | Analytical views (`v_latest_rates`, `v_forecast_summary`) for fast ad-hoc SQL |

---

## Key Engineering Decisions

- **PySpark for transformations** - window functions (LAG, rolling avg, stddev) are expressed as relational algebra, making them easy to test and scale
- **Explicit schemas at every layer** - `src/utils/schema.py` declares all StructTypes centrally; schema drift is caught at write time, not query time
- **Idempotent writes** - every pipeline stage can be re-run safely; raw records are upserted by date, Parquet layers use `mode="overwrite"`
- **DuckDB as the serving layer** - reads Parquet natively without copying data; opened and closed per query so the pipeline can always acquire the write lock
- **Config-driven** - currencies, windows, thresholds, and paths all live in `config/settings.yaml`; no magic values in source code
- **Pure transformation functions** - each transform (e.g. `add_moving_averages`, `compute_daily_returns`) is a standalone function that takes and returns a DataFrame, making unit testing straightforward

### Scope and tradeoffs

A few choices are deliberate for a portfolio project and would differ in a high-scale production system:

- **Spark demonstrates the API, not a data-size need.** The dataset is a few currency pairs at daily granularity (well under 50 MB), which DuckDB or pandas would handle comfortably. The transforms are written as reusable, tested window functions so the logic would scale out, but at this volume Spark is the heavier option by choice. DuckDB already serves the same data on the read side.
- **Forecasts are backtested, not trusted blindly.** FX rates are close to a random walk, so the ML stage reports walk-forward MAPE and 95% interval coverage per pair (surfaced in the Forecasts tab) rather than presenting predictions as fact. In practice the backtest shows the 95% intervals under-cover badly at a 30-day horizon (coverage well below 95%), so the point forecasts are indicative and the bands are optimistic.
- **The daily job commits data back to the repo.** This keeps the hosted Streamlit demo self-contained and free, at the cost of data churn in git history. A production deployment would publish to object storage or a warehouse instead.

---

## Dashboard

| Tab | Contents |
|---|---|
| 📈 Historical Rates | Line chart of exchange rates over time |
| 📊 Daily Returns | Bar chart of % daily changes |
| 🔬 Analytics | Rolling averages & volatility heatmap |
| 🔮 Forecasts | 30-day Prophet forecast with 95% CI band |
| 🐥 DuckDB Explorer | Ad-hoc SQL against the warehouse |

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/pipeline.yml`) runs:

- **On every push / PR** - `ruff` lint + full unit test suite with coverage
- **Weekdays at 06:00 UTC** - full pipeline, then commits updated gold/forecasts/warehouse back to the repo, triggering a Streamlit Cloud redeploy

---

## Project Structure

```
currency_intelligence/
├── config/
│   ├── settings.yaml              # Single source of truth for all config
│   └── config_loader.py           # Typed dataclass config loader
├── src/
│   ├── utils/
│   │   ├── schema.py              # PySpark schemas for all layers
│   │   └── spark_utils.py         # Reusable Spark helpers (read, write, dedup)
│   ├── ingestion/fetch_rates.py   # HTTP fetch + JSONL upsert
│   ├── bronze/raw_to_bronze.py    # JSONL -> structured Parquet
│   ├── silver/bronze_to_silver.py # Validation, pairs, daily returns
│   ├── gold/silver_to_gold.py     # Rolling stats, volatility, z-scores
│   ├── ml/forecast.py             # Prophet training + prediction
│   └── warehouse/load_duckdb.py   # Load Parquet into DuckDB, create views
├── dashboard/
│   └── app.py                     # Streamlit multi-tab dashboard
├── tests/
│   ├── conftest.py                # Shared SparkSession fixture (session-scoped)
│   ├── test_ingestion.py
│   ├── test_bronze.py
│   ├── test_silver.py
│   └── test_gold.py
├── .github/workflows/pipeline.yml
├── pyproject.toml                 # Dependencies and dev group (uv)
├── uv.lock                        # Fully pinned, cross-platform lockfile
└── run_pipeline.py                # Pipeline CLI orchestrator
```

---

## Quickstart

**Prerequisites:** [uv](https://docs.astral.sh/uv/), Java 17+ (required by PySpark)

```bash
git clone https://github.com/davronp/currency_intelligence.git
cd currency_intelligence
uv sync --extra pipeline

# Backfill 90 days of history (recommended for Prophet to have enough training data)
uv run python run_pipeline.py --backfill-days 90

# Launch the dashboard
uv run streamlit run dashboard/app.py
```

`uv sync` installs Python 3.12 (pinned in `.python-version`) and every dependency from `uv.lock`; no manual venv needed. The base install is the dashboard's runtime; the `pipeline` extra adds the heavy Spark/Prophet stack, so pass `--extra pipeline` to run the ingestion and forecasting pipeline locally.

### Pipeline CLI

```bash
# Today only
uv run python run_pipeline.py

# Explicit date range
uv run python run_pipeline.py --start-date 2025-01-01 --end-date 2025-03-01

# Re-fetch already-ingested dates (e.g. after adding a new currency)
uv run python run_pipeline.py --backfill-days 90 --force-ingest

# Skip ML forecasting
uv run python run_pipeline.py --skip-ml

# Run only specific stages
uv run python run_pipeline.py --stages silver gold warehouse
```

### Tests

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

### Add a new currency

Edit `config/settings.yaml` -> `currencies.targets`, then backfill:

```bash
uv run python run_pipeline.py --backfill-days 90 --force-ingest
```
