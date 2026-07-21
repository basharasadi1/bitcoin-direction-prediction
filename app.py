from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitcoin_direction.config import ARTIFACTS_DIR, DATA_PATH
from bitcoin_direction.features import load_raw_data
from bitcoin_direction.inference import predict_latest


st.set_page_config(page_title="Bitcoin Direction ML", page_icon="₿", layout="wide")
st.title("Bitcoin Direction Prediction")
st.caption("Academic machine-learning portfolio project — not financial advice.")

metadata_path = ARTIFACTS_DIR / "metadata.json"
if not metadata_path.exists():
    st.error("Artifacts are missing. Run `python train.py` first.")
    st.stop()

with metadata_path.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

model_results = pd.read_csv(ARTIFACTS_DIR / "results" / "model_results.csv")
backtest_summary = pd.read_csv(ARTIFACTS_DIR / "results" / "backtest_summary.csv")
raw = load_raw_data(str(DATA_PATH))
latest_predictions = predict_latest(DATA_PATH)

st.subheader("Latest model outputs")
columns = st.columns(3)
for column, row in zip(columns, latest_predictions.itertuples(index=False)):
    with column:
        st.metric(
            label=row.horizon.replace("_", " ").title(),
            value=row.prediction,
            delta=f"P(UP) {row.probability_up:.1%} | threshold {row.threshold:.1%}",
            delta_color="off",
        )
        st.caption(f"Model: {row.model} · Data through {row.data_date}")

st.warning(
    "The interleaved test weeks do not demonstrate a stable predictive edge. Treat these probabilities as model outputs, not trading signals."
)

overview_tab, comparison_tab, backtest_tab, data_tab, upload_tab = st.tabs(
    ["Overview", "Model comparison", "Backtests", "Dataset", "Predict new CSV"]
)

with overview_tab:
    price_chart = px.line(raw, x="date", y="close", title="BTC/USDT daily close")
    price_chart.update_layout(xaxis_title="Date", yaxis_title="USDT")
    st.plotly_chart(price_chart, use_container_width=True)

    split = metadata["split"]
    info = pd.DataFrame(
        {
            "Item": [
                "Raw rows",
                "Model rows",
                "Features",
                "Split strategy",
                "Training weeks",
                "Validation weeks",
                "Test weeks",
                "Rows by split",
            ],
            "Value": [
                split["raw_rows"],
                split["model_rows"],
                split["feature_count"],
                "Repeating 10-week cycle across full timeline",
                str(split["train_cycle_weeks"]),
                str(split["validation_cycle_weeks"]),
                str(split["test_cycle_weeks"]),
                f"{split['train_rows']} / {split['validation_rows']} / {split['test_rows']}",
            ],
        }
    )
    st.dataframe(info, use_container_width=True, hide_index=True)

with comparison_tab:
    split_choice = st.radio("Evaluation split", ["validation", "test"], horizontal=True)
    metric_choice = st.selectbox(
        "Metric", ["roc_auc", "balanced_accuracy", "accuracy", "f1", "precision", "recall"]
    )
    filtered = model_results[model_results["split"] == split_choice].copy()
    figure = px.bar(
        filtered,
        x="model",
        y=metric_choice,
        color="horizon_name",
        barmode="group",
        title=f"{split_choice.title()} {metric_choice.replace('_', ' ').title()}",
    )
    figure.update_yaxes(range=[0, 1])
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        filtered[
            [
                "horizon_name",
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "threshold",
            ]
        ].sort_values(["horizon_name", "roc_auc"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )

    walk_forward_path = ARTIFACTS_DIR / "results" / "walk_forward_summary.csv"
    if walk_forward_path.exists():
        st.subheader("Classical-model walk-forward validation")
        walk_forward = pd.read_csv(walk_forward_path)
        st.dataframe(walk_forward, use_container_width=True, hide_index=True)

with backtest_tab:
    st.caption("Non-overlapping horizon windows, 0.1% fee per entry and exit, cash when the model predicts DOWN.")
    horizon_name = st.selectbox("Horizon", backtest_summary["horizon_name"].tolist())
    backtest = pd.read_csv(ARTIFACTS_DIR / "backtests" / f"backtest_{horizon_name}.csv")
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=backtest["date"], y=backtest["strategy_equity"], name="Model strategy"))
    figure.add_trace(go.Scatter(x=backtest["date"], y=backtest["buy_hold_equity"], name="Buy and hold"))
    figure.update_layout(title=f"{horizon_name.replace('_', ' ').title()} backtest", xaxis_title="Date", yaxis_title="Value of 1 USDT")
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        backtest_summary[backtest_summary["horizon_name"] == horizon_name],
        use_container_width=True,
        hide_index=True,
    )

with data_tab:
    st.write(f"Rows: {len(raw):,} · Start: {raw['date'].min().date()} · End: {raw['date'].max().date()}")
    st.dataframe(raw.tail(100), use_container_width=True, hide_index=True)

with upload_tab:
    st.write("Upload a newer daily Binance-format CSV with the same columns.")
    uploaded = st.file_uploader("Daily BTC/USDT CSV", type=["csv"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temporary_file:
            temporary_file.write(uploaded.getbuffer())
            temporary_path = temporary_file.name
        try:
            uploaded_predictions = predict_latest(temporary_path)
            st.success("Prediction completed.")
            st.dataframe(uploaded_predictions, use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"Could not process the CSV: {error}")
        finally:
            Path(temporary_path).unlink(missing_ok=True)
