# 💱 Currency Intelligence Platform

A production-grade data engineering platform that ingests live exchange rates for **USD/EUR**, **USD/SEK**, and **USD/UZS**, runs a full **Medallion architecture** data pipeline, trains **Prophet time-series forecasts**, loads curated data into a **DuckDB warehouse**, and serves everything through an interactive **Streamlit dashboard**.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Ingestion   │ ->  │    Bronze    │ -> │      Silver  │ -> │       Gold   │
│  (Raw JSON)  │     │  (Parquet)   │    │  (Parquet)   │    │  (Parquet)   │
└──────────────┘     └──────────────┘    └──────────────┘    └──────────────┘
                                                                      │
                                              ┌───────────────────────┤
                                              ▼                       ▼
                                    ┌──────────────────┐  ┌─────────────────┐
                                    │  ML Forecasting  │  │  DuckDB         │
                                    │  (Prophet)       │  │  Warehouse      │
                                    └──────────────────┘  └─────────────────┘
                                              │                       │
                                              └───────────┬───────────┘
                                                          ▼
                                               ┌─────────────────────┐
                                               │  Streamlit Dashboard│
                                               └─────────────────────┘
```

### Medallion Layers

| Layer | Format | Contents |
|---|---|---|
| **Raw** | JSON | Verbatim API responses, date-partitioned |
| **Bronze** | Parquet | Parsed, deduplicated, schema-enforced |
| **Silver** | Parquet | Validated, currency pairs, daily returns |
| **Gold** | Parquet | MA-7/30/90, volatility, z-scores |
| **Forecasts** | Parquet | 30-day Prophet predictions per pair |
| **Warehouse** | DuckDB | Analytical views for fast queries |

---

## Project Structure

```
currency_intelligence/
├── config/
│   ├── settings.yaml          # All configuration (no hardcoded values)
│   └── config_loader.py       # Typed dataclass config loader
├── src/
│   ├── utils/
│   │   ├── logger.py          # Centralised logging factory
│   │   ├── schema.py          # Explicit Spark schemas for all layers
│   │   └── spark_utils.py     # Reusable Spark transformation helpers
│   ├── ingestion/
│   │   └── fetch_rates.py     # HTTP fetch + raw JSON persistence
│   ├── bronze/
│   │   └── raw_to_bronze.py   # JSON → structured Parquet
│   ├── silver/
│   │   └── bronze_to_silver.py # Clean, normalise, daily returns
│   ├── gold/
│   │   └── silver_to_gold.py  # Rolling averages, volatility, z-scores
│   ├── ml/
│   │   └── forecast.py        # Prophet training + prediction
│   └── warehouse/
│       └── load_duckdb.py     # Load gold + forecasts into DuckDB
├── dashboard/
│   └── app.py                 # Streamlit multi-tab dashboard
├── tests/
│   ├── conftest.py            # Shared SparkSession fixture
│   ├── test_ingestion.py
│   ├── test_bronze.py
│   ├── test_silver.py
│   └── test_gold.py
├── .github/
│   └── workflows/
│       └── pipeline.yml       # CI: lint → test → daily pipeline
├── data/                      # Auto-created; gitignored
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   ├── forecasts/
│   └── warehouse/
├── requirements.txt
├── run_pipeline.py            # Main orchestrator CLI
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- Java 11+ (required by PySpark)

```bash
# 1. Clone & enter the project
git clone <repo-url>
cd currency_intelligence

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (fetches live data)
python run_pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

---

## Running Individual Stages

```bash
# Run only specific stages
python run_pipeline.py --stages ingestion bronze silver

# Run for a historical date
python run_pipeline.py --date 2024-01-15

# Skip ML forecasting (faster)
python run_pipeline.py --skip-ml

# Verbose logging
python run_pipeline.py --log-level DEBUG
```

---

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_silver.py -v
```

---

## Dashboard Tabs

| Tab | Contents |
|---|---|
| 📈 Historical Rates | Line chart of exchange rates over time |
| 📊 Daily Returns | Bar chart of % daily changes |
| 🔬 Analytics | Moving averages & volatility heatmap |
| 🔮 Forecasts | 30-day Prophet forecast with 95% CI band |
| 🐥 DuckDB Explorer | Ad-hoc SQL against the warehouse |

---

## Configuration

All parameters are in `config/settings.yaml`. Key settings:

```yaml
currencies:
  base: USD
  targets: [EUR, SEK, UZS]   # Add or remove currencies here

gold:
  rolling_windows: [7, 30, 90]
  volatility_window: 30

ml:
  forecast_horizon_days: 30
  min_training_rows: 30
```

---

## Design Principles

- **Idempotent writes** - all Parquet writes use `mode="overwrite"` on the target partition
- **Explicit schemas** - every layer's schema is declared in `src/utils/schema.py`
- **Pure transformation functions** - each stage function is independently testable
- **Separation of concerns** - ingestion, transform, write, and orchestration are separate functions
- **Structured logging** - every module uses the centralised logger; no `print()` statements
- **Config-driven** - no magic strings or hardcoded paths anywhere in the source

---

## CI/CD (GitHub Actions)

The workflow (`.github/workflows/pipeline.yml`) runs:

1. **On every push/PR** → lint + unit tests
2. **Daily at 06:00 UTC** → full pipeline (ingestion through warehouse)
3. **Manual dispatch** → pipeline with optional date override and skip-ML flag

---

## Extending the Platform

**Add a new currency:**
Edit `config/settings.yaml` → `currencies.targets`

**Add a new gold feature:**
Add a pure function to `src/gold/silver_to_gold.py`, update `GOLD_SCHEMA` in `src/utils/schema.py`, add a unit test in `tests/test_gold.py`

**Change forecast horizon:**
Edit `config/settings.yaml` → `ml.forecast_horizon_days`

**Add a new DuckDB view:**
Add a `CREATE OR REPLACE VIEW` statement in `src/warehouse/load_duckdb.py` → `create_analytical_views()`
