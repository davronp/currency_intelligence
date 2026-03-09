"""dashboard/app.py

Streamlit dashboard for the Currency Intelligence Platform.

Tabs
----
1. Historical Rates   - time-series chart of exchange rates
2. Daily Returns      - bar chart of % daily returns
3. Analytics          - rolling averages and volatility heatmap
4. Forecasts          - 30-day Prophet forecast with confidence band
5. DuckDB Explorer    - run ad-hoc SQL against the warehouse

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from config.config_loader import load_config

st.set_page_config(
    page_title="Currency Intelligence",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

cfg = load_config()


st.sidebar.title("💱 Currency Intelligence")
st.sidebar.caption(f"v{cfg.project.version}")

available_pairs = cfg.currencies.pairs

selected_pairs = st.sidebar.multiselect(
    "Currency Pairs",
    options=available_pairs,
    default=available_pairs[:2],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Data layers: **Raw → Bronze → Silver → Gold**  \nForecasting: **Prophet**  \nWarehouse: **DuckDB**"
)


@st.cache_data(ttl=300)
def load_gold() -> pd.DataFrame | None:
    gold_path = cfg.paths.gold
    if not gold_path.exists():
        return None
    try:
        df = pd.read_parquet(gold_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    except Exception as e:
        st.error(f"Could not load gold data: {e}")
        return None


@st.cache_data(ttl=300)
def load_forecasts() -> pd.DataFrame | None:
    fc_path = cfg.paths.forecasts / "forecasts.parquet"
    if not fc_path.exists():
        return None
    try:
        df = pd.read_parquet(fc_path)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        return df
    except Exception as e:
        st.error(f"Could not load forecast data: {e}")
        return None


@st.cache_resource
def get_duckdb_conn():
    db_path = cfg.warehouse.db_file
    if not db_path.exists():
        return None
    try:
        import duckdb

        return duckdb.connect(str(db_path), read_only=True)
    except Exception:
        return None


st.title("💱 Currency Intelligence Platform")

df_gold = load_gold()
df_fc = load_forecasts()

if df_gold is None:
    st.warning("⚠️  No gold data found.  Run the full pipeline first:\n\n```bash\npython run_pipeline.py\n```")
    st.stop()

df_view = df_gold[df_gold["currency_pair"].isin(selected_pairs)] if selected_pairs else df_gold


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Historical Rates",
        "📊 Daily Returns",
        "🔬 Analytics",
        "🔮 Forecasts",
        "🐥 DuckDB Explorer",
    ]
)


with tab1:
    st.subheader("Historical Exchange Rates")
    st.caption("Base currency: USD")

    if df_view.empty:
        st.info("Select at least one currency pair from the sidebar.")
    else:
        import plotly.express as px

        fig = px.line(
            df_view,
            x="date",
            y="rate",
            color="currency_pair",
            labels={"rate": "Rate (USD base)", "date": "Date"},
            template="plotly_dark",
        )
        fig.update_layout(legend_title_text="Pair", height=450)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for i, pair in enumerate(selected_pairs[:3]):
            subset = df_view[df_view["currency_pair"] == pair]
            if not subset.empty:
                latest = subset.iloc[-1]
                with [col1, col2, col3][i]:
                    prev = subset.iloc[-2]["rate"] if len(subset) > 1 else latest["rate"]
                    delta = ((latest["rate"] - prev) / prev) * 100
                    st.metric(
                        label=pair,
                        value=f"{latest['rate']:.4f}",
                        delta=f"{delta:+.3f}%",
                    )


with tab2:
    st.subheader("Daily Returns (%)")

    if df_view.empty or "daily_return" not in df_view.columns:
        st.info("Daily return data not available.")
    else:
        import plotly.express as px

        df_ret = df_view[df_view["daily_return"].notna()].copy()
        df_ret["daily_return_pct"] = df_ret["daily_return"] * 100

        fig = px.bar(
            df_ret,
            x="date",
            y="daily_return_pct",
            color="currency_pair",
            barmode="group",
            labels={"daily_return_pct": "Daily Return (%)", "date": "Date"},
            template="plotly_dark",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.markdown("**Return statistics**")
        stats = (
            df_ret.groupby("currency_pair")["daily_return_pct"].agg(["mean", "std", "min", "max"]).round(4)
        )
        stats.columns = ["Mean %", "Std Dev %", "Min %", "Max %"]
        st.dataframe(stats, use_container_width=True)


with tab3:
    st.subheader("Rolling Averages & Volatility")

    if df_view.empty:
        st.info("No data to display.")
    else:
        import plotly.graph_objects as go

        for pair in selected_pairs:
            pair_df = df_view[df_view["currency_pair"] == pair]
            if pair_df.empty:
                continue

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pair_df["date"],
                    y=pair_df["rate"],
                    name="Rate",
                    line={"width": 1},
                )
            )
            for ma_col, colour in [
                ("ma_7", "#f59e0b"),
                ("ma_30", "#10b981"),
                ("ma_90", "#6366f1"),
            ]:
                if ma_col in pair_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pair_df["date"],
                            y=pair_df[ma_col],
                            name=ma_col.replace("_", " ").upper(),
                            line={"dash": "dot", "color": colour},
                        )
                    )
            fig.update_layout(
                title=f"{pair} — Moving Averages",
                template="plotly_dark",
                height=350,
                legend={"orientation": "h"},
            )
            st.plotly_chart(fig, use_container_width=True)

        # Volatility heatmap
        st.markdown("**30-day Volatility (std dev of daily returns)**")
        if "volatility_30" in df_view.columns:
            import plotly.express as px

            vol_df = df_view[df_view["volatility_30"].notna()].copy()
            vol_pivot = vol_df.pivot_table(index="date", columns="currency_pair", values="volatility_30")
            fig_heat = px.imshow(
                vol_pivot.T,
                labels={"color": "Volatility"},
                template="plotly_dark",
                color_continuous_scale="Reds",
                aspect="auto",
            )
            fig_heat.update_layout(height=250)
            st.plotly_chart(fig_heat, use_container_width=True)


with tab4:
    st.subheader("30-Day Prophet Forecasts")

    if df_fc is None:
        st.info("No forecast data found.  Run `python run_pipeline.py` to generate forecasts.")
    else:
        import plotly.graph_objects as go

        fc_pairs = [p for p in selected_pairs if p in df_fc["currency_pair"].unique()]
        if not fc_pairs:
            st.info("No forecast data for the selected pairs.")

        for pair in fc_pairs:
            fc_pair = df_fc[df_fc["currency_pair"] == pair].sort_values("forecast_date")
            hist_pair = df_view[df_view["currency_pair"] == pair].tail(90)

            fig = go.Figure()

            # Historical
            fig.add_trace(
                go.Scatter(
                    x=hist_pair["date"],
                    y=hist_pair["rate"],
                    name="Historical",
                    line={"color": "#60a5fa", "width": 1.5},
                )
            )

            # Forecast with confidence band
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([fc_pair["forecast_date"], fc_pair["forecast_date"].iloc[::-1]]),
                    y=pd.concat([fc_pair["yhat_upper"], fc_pair["yhat_lower"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(167, 139, 250, 0.2)",
                    line={"color": "rgba(255,255,255,0)"},
                    name="95% CI",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc_pair["forecast_date"],
                    y=fc_pair["yhat"],
                    name="Forecast",
                    line={"color": "#a78bfa", "width": 2, "dash": "dash"},
                )
            )

            fig.update_layout(
                title=f"{pair} — 30-Day Forecast",
                template="plotly_dark",
                height=380,
                legend={"orientation": "h"},
                xaxis_title="Date",
                yaxis_title="Rate",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            with st.expander(f"View {pair} forecast data"):
                st.dataframe(
                    fc_pair[["forecast_date", "yhat", "yhat_lower", "yhat_upper"]]
                    .rename(
                        columns={
                            "forecast_date": "Date",
                            "yhat": "Forecast",
                            "yhat_lower": "Lower (95%)",
                            "yhat_upper": "Upper (95%)",
                        }
                    )
                    .set_index("Date")
                    .round(5),
                    use_container_width=True,
                )


with tab5:
    st.subheader("🐥 DuckDB SQL Explorer")

    conn = get_duckdb_conn()
    if conn is None:
        st.info(
            "DuckDB warehouse not found.  Run the full pipeline first:\n\n"
            "```bash\npython run_pipeline.py\n```"
        )
    else:
        example_queries = {
            "Latest rates": "SELECT * FROM v_latest_rates;",
            "Forecast summary": "SELECT * FROM v_forecast_summary LIMIT 30;",
            "Most volatile pair": """
SELECT currency_pair, AVG(volatility_30) AS avg_vol
FROM gold_exchange_rates
WHERE volatility_30 IS NOT NULL
GROUP BY 1
ORDER BY 2 DESC;""",
            "Rate z-score outliers": """
SELECT date, currency_pair, rate, rate_z_score
FROM gold_exchange_rates
WHERE ABS(rate_z_score) > 2
ORDER BY ABS(rate_z_score) DESC
LIMIT 20;""",
        }

        selected_example = st.selectbox("Load example query", ["Custom…", *list(example_queries.keys())])
        default_sql = example_queries.get(selected_example, "SELECT * FROM gold_exchange_rates LIMIT 10;")

        user_sql = st.text_area("SQL", value=default_sql, height=140)

        if st.button("▶ Run Query", type="primary"):
            try:
                result = conn.execute(user_sql).df()
                st.success(f"{len(result)} row(s) returned")
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")

        st.markdown("**Available tables & views**")
        try:
            tables = conn.execute(
                "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema='main' ORDER BY 1"
            ).df()
            st.dataframe(tables, use_container_width=True)
        except Exception:
            pass
