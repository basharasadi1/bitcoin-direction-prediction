from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import ARTIFACTS_DIR, CONFIG, DATA_PATH
from .features import create_feature_table, repeating_week_split_indices, load_raw_data
from .metrics import choose_threshold, classification_metrics
from .torch_models import (
    LSTMClassifierNet,
    MLPClassifierNet,
    create_sequences,
    predict_probabilities,
    train_binary_model,
    train_fixed_epochs,
)


def ensure_directories() -> None:
    for directory in (
        ARTIFACTS_DIR / "models",
        ARTIFACTS_DIR / "results",
        ARTIFACTS_DIR / "predictions",
        ARTIFACTS_DIR / "backtests",
        ARTIFACTS_DIR / "figures",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def sklearn_models() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=CONFIG.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=7,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=CONFIG.random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=5,
            reg_alpha=0.05,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=CONFIG.random_state,
            n_jobs=4,
            tree_method="hist",
        ),
    }


def _safe_roc_auc(metrics: dict[str, Any]) -> float:
    value = float(metrics.get("roc_auc", float("nan")))
    return value if math.isfinite(value) else -1.0


def _prediction_frame(
    df: pd.DataFrame,
    row_indices: np.ndarray,
    horizon: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": row_indices,
            "date": df.iloc[row_indices]["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "close": df.iloc[row_indices]["close"].to_numpy(),
            "future_return": df.iloc[row_indices][f"future_return_{horizon}d"].to_numpy(),
            "actual_direction": df.iloc[row_indices][f"target_{horizon}d"].to_numpy(),
        }
    )


def _maximum_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def create_non_overlapping_backtest(
    predictions: pd.DataFrame,
    probability_column: str,
    threshold: float,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = predictions.dropna(subset=[probability_column]).sort_values("date").copy()
    ordered["date"] = pd.to_datetime(ordered["date"], utc=True)
    selected_rows: list[int] = []
    next_available_date: pd.Timestamp | None = None
    for row_index, row in ordered.iterrows():
        current_date = row["date"]
        if next_available_date is None or current_date >= next_available_date:
            selected_rows.append(row_index)
            next_available_date = current_date + pd.Timedelta(days=horizon)
    selected = ordered.loc[selected_rows].copy().reset_index(drop=True)
    selected["date"] = selected["date"].dt.strftime("%Y-%m-%d")
    selected["signal"] = (selected[probability_column] >= threshold).astype(int)
    selected["strategy_return"] = (
        selected["signal"] * selected["future_return"]
        - selected["signal"] * 2 * CONFIG.transaction_fee_per_side
    )
    selected["buy_hold_return"] = selected["future_return"]
    selected["strategy_equity"] = (1 + selected["strategy_return"]).cumprod()
    selected["buy_hold_equity"] = (1 + selected["buy_hold_return"]).cumprod()

    periods_per_year = 365 / horizon
    strategy_std = selected["strategy_return"].std(ddof=0)
    sharpe = (
        float(selected["strategy_return"].mean() / strategy_std * np.sqrt(periods_per_year))
        if strategy_std > 0
        else 0.0
    )

    summary = {
        "strategy_final_equity": float(selected["strategy_equity"].iloc[-1]),
        "buy_hold_final_equity": float(selected["buy_hold_equity"].iloc[-1]),
        "strategy_total_return": float(selected["strategy_equity"].iloc[-1] - 1),
        "buy_hold_total_return": float(selected["buy_hold_equity"].iloc[-1] - 1),
        "strategy_max_drawdown": _maximum_drawdown(selected["strategy_equity"]),
        "buy_hold_max_drawdown": _maximum_drawdown(selected["buy_hold_equity"]),
        "strategy_sharpe": sharpe,
        "number_of_periods": int(len(selected)),
        "invested_fraction": float(selected["signal"].mean()),
    }
    return selected, summary


def _plot_overview(raw: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(raw["date"], raw["close"])
    plt.title("BTC/USDT Daily Closing Price")
    plt.xlabel("Date")
    plt.ylabel("USDT")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "figures" / "price_history.png", dpi=160)
    plt.close()

    balances = []
    for horizon in CONFIG.horizons:
        balance = feature_df[f"target_{horizon}d"].value_counts(normalize=True).sort_index()
        balances.append(
            {
                "horizon": CONFIG.horizon_names[horizon],
                "down": float(balance.get(0, 0.0)),
                "up": float(balance.get(1, 0.0)),
            }
        )
    pd.DataFrame(balances).to_csv(
        ARTIFACTS_DIR / "results" / "class_balance.csv", index=False
    )


def _plot_model_comparison(results: pd.DataFrame) -> None:
    test_results = results[results["split"] == "test"].copy()
    if test_results.empty:
        return
    for metric in ("roc_auc", "balanced_accuracy", "f1"):
        pivot = test_results.pivot(index="model", columns="horizon_name", values=metric)
        ax = pivot.plot(kind="bar", figsize=(11, 6))
        ax.set_title(f"Test {metric.replace('_', ' ').title()} by Model")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "figures" / f"test_{metric}.png", dpi=160)
        plt.close()


def _plot_backtest(backtest: pd.DataFrame, horizon: int, model_name: str) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(pd.to_datetime(backtest["date"]), backtest["strategy_equity"], label="Model strategy")
    plt.plot(pd.to_datetime(backtest["date"]), backtest["buy_hold_equity"], label="Buy and hold")
    plt.title(f"{CONFIG.horizon_names[horizon]}: {model_name} backtest")
    plt.xlabel("Date")
    plt.ylabel("Value of 1 USDT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        ARTIFACTS_DIR / "figures" / f"backtest_{CONFIG.horizon_names[horizon]}.png",
        dpi=160,
    )
    plt.close()


def train_project(data_path: str | Path = DATA_PATH) -> dict[str, Any]:
    ensure_directories()
    np.random.seed(CONFIG.random_state)
    torch.manual_seed(CONFIG.random_state)

    raw = load_raw_data(str(data_path))
    feature_df, feature_columns = create_feature_table(raw, drop_unlabeled=True)
    latest_feature_df, latest_feature_columns = create_feature_table(raw, drop_unlabeled=False)
    if latest_feature_columns != feature_columns:
        raise RuntimeError("Feature columns differ between labeled and latest datasets.")
    train_idx, validation_idx, test_idx, split_assignments = repeating_week_split_indices(
        feature_df["date"], anchor_date=raw["date"].min()
    )
    feature_df = pd.concat([feature_df, split_assignments], axis=1)
    feature_df.to_csv(ARTIFACTS_DIR / "results" / "engineered_dataset.csv", index=False)

    week_anchor = raw["date"].min().normalize() - pd.Timedelta(
        days=int(raw["date"].min().dayofweek)
    )
    split_info = {
        "strategy": "repeating_10_week_interleaved",
        "description": "Monday-Sunday weeks repeated in a 10-week cycle across the full timeline.",
        "raw_rows": int(len(raw)),
        "model_rows": int(len(feature_df)),
        "feature_count": int(len(feature_columns)),
        "raw_start": raw["date"].min().isoformat(),
        "raw_end": raw["date"].max().isoformat(),
        "week_anchor": week_anchor.isoformat(),
        "cycle_length_weeks": CONFIG.split_cycle_length_weeks,
        "train_cycle_weeks": list(CONFIG.train_cycle_weeks),
        "validation_cycle_weeks": list(CONFIG.validation_cycle_weeks),
        "test_cycle_weeks": list(CONFIG.test_cycle_weeks),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "test_rows": int(len(test_idx)),
        "train_fraction": float(len(train_idx) / len(feature_df)),
        "validation_fraction": float(len(validation_idx) / len(feature_df)),
        "test_fraction": float(len(test_idx) / len(feature_df)),
        "train_first_date": feature_df.iloc[train_idx[0]]["date"].isoformat(),
        "train_last_date": feature_df.iloc[train_idx[-1]]["date"].isoformat(),
        "validation_first_date": feature_df.iloc[validation_idx[0]]["date"].isoformat(),
        "validation_last_date": feature_df.iloc[validation_idx[-1]]["date"].isoformat(),
        "test_first_date": feature_df.iloc[test_idx[0]]["date"].isoformat(),
        "test_last_date": feature_df.iloc[test_idx[-1]]["date"].isoformat(),
    }
    _plot_overview(raw, feature_df)

    x = feature_df[feature_columns].to_numpy(dtype=np.float64)
    all_result_rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []
    horizon_metadata: dict[str, Any] = {}

    for horizon in CONFIG.horizons:
        horizon_name = CONFIG.horizon_names[horizon]
        target_column = f"target_{horizon}d"
        y = feature_df[target_column].to_numpy(dtype=int)

        x_train, y_train = x[train_idx], y[train_idx]
        x_validation, y_validation = x[validation_idx], y[validation_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        validation_predictions = _prediction_frame(feature_df, validation_idx, horizon)
        test_predictions = _prediction_frame(feature_df, test_idx, horizon)
        model_objects: dict[str, Any] = {}
        model_thresholds: dict[str, float] = {}
        model_validation_auc: dict[str, float] = {}

        # Majority baseline.
        majority_class = int(np.mean(y_train) >= 0.5)
        baseline_probability = float(np.mean(y_train))
        baseline_val_prob = np.full(len(y_validation), baseline_probability)
        baseline_test_prob = np.full(len(y_test), baseline_probability)
        baseline_threshold = 0.5
        validation_predictions["prob_majority_baseline"] = baseline_val_prob
        test_predictions["prob_majority_baseline"] = baseline_test_prob
        for split_name, y_true, probabilities in (
            ("validation", y_validation, baseline_val_prob),
            ("test", y_test, baseline_test_prob),
        ):
            metric_row = classification_metrics(y_true, probabilities, baseline_threshold)
            metric_row.update(
                {
                    "horizon_days": horizon,
                    "horizon_name": horizon_name,
                    "model": "majority_baseline",
                    "split": split_name,
                    "majority_class": majority_class,
                }
            )
            all_result_rows.append(metric_row)

        # Scikit-learn models.
        for model_name, base_model in sklearn_models().items():
            model = clone(base_model)
            model.fit(x_train, y_train)
            validation_prob = model.predict_proba(x_validation)[:, 1]
            threshold = choose_threshold(y_validation, validation_prob)
            test_prob = model.predict_proba(x_test)[:, 1]

            validation_predictions[f"prob_{model_name}"] = validation_prob
            test_predictions[f"prob_{model_name}"] = test_prob
            model_objects[model_name] = model
            model_thresholds[model_name] = threshold

            for split_name, y_true, probabilities in (
                ("validation", y_validation, validation_prob),
                ("test", y_test, test_prob),
            ):
                metric_row = classification_metrics(y_true, probabilities, threshold)
                metric_row.update(
                    {
                        "horizon_days": horizon,
                        "horizon_name": horizon_name,
                        "model": model_name,
                        "split": split_name,
                    }
                )
                all_result_rows.append(metric_row)
                if split_name == "validation":
                    model_validation_auc[model_name] = _safe_roc_auc(metric_row)

            joblib.dump(
                model,
                ARTIFACTS_DIR / "models" / f"{horizon_name}_{model_name}.joblib",
            )

        # Shared scaler for PyTorch models, fitted on training data only.
        neural_scaler = StandardScaler().fit(x_train)
        x_train_scaled = neural_scaler.transform(x_train).astype(np.float32)
        x_validation_scaled = neural_scaler.transform(x_validation).astype(np.float32)
        x_test_scaled = neural_scaler.transform(x_test).astype(np.float32)
        joblib.dump(
            neural_scaler,
            ARTIFACTS_DIR / "models" / f"{horizon_name}_neural_scaler.joblib",
        )

        # Feedforward neural network.
        mlp = MLPClassifierNet(input_size=len(feature_columns))
        mlp, mlp_history = train_binary_model(
            mlp,
            x_train_scaled,
            y_train,
            x_validation_scaled,
            y_validation,
        )
        mlp_val_prob = predict_probabilities(mlp, x_validation_scaled)
        mlp_threshold = choose_threshold(y_validation, mlp_val_prob)
        mlp_test_prob = predict_probabilities(mlp, x_test_scaled)
        validation_predictions["prob_mlp"] = mlp_val_prob
        test_predictions["prob_mlp"] = mlp_test_prob
        model_thresholds["mlp"] = mlp_threshold
        model_objects["mlp"] = mlp

        for split_name, y_true, probabilities in (
            ("validation", y_validation, mlp_val_prob),
            ("test", y_test, mlp_test_prob),
        ):
            metric_row = classification_metrics(y_true, probabilities, mlp_threshold)
            metric_row.update(
                {
                    "horizon_days": horizon,
                    "horizon_name": horizon_name,
                    "model": "mlp",
                    "split": split_name,
                    "best_epoch": mlp_history.best_epoch,
                }
            )
            all_result_rows.append(metric_row)
            if split_name == "validation":
                model_validation_auc["mlp"] = _safe_roc_auc(metric_row)

        torch.save(
            {
                "state_dict": mlp.state_dict(),
                "input_size": len(feature_columns),
            },
            ARTIFACTS_DIR / "models" / f"{horizon_name}_mlp.pt",
        )

        # LSTM sequences are created across the complete timeline, then split by target row.
        x_all_scaled = neural_scaler.transform(x).astype(np.float32)
        x_sequences, y_sequences, sequence_target_indices = create_sequences(
            x_all_scaled,
            y,
            CONFIG.sequence_length,
        )
        sequence_train_mask = np.isin(sequence_target_indices, train_idx)
        sequence_validation_mask = np.isin(sequence_target_indices, validation_idx)
        sequence_test_mask = np.isin(sequence_target_indices, test_idx)

        lstm = LSTMClassifierNet(input_size=len(feature_columns))
        lstm, lstm_history = train_binary_model(
            lstm,
            x_sequences[sequence_train_mask],
            y_sequences[sequence_train_mask],
            x_sequences[sequence_validation_mask],
            y_sequences[sequence_validation_mask],
            learning_rate=8e-4,
        )
        lstm_val_prob = predict_probabilities(lstm, x_sequences[sequence_validation_mask])
        lstm_threshold = choose_threshold(y_sequences[sequence_validation_mask], lstm_val_prob)
        lstm_test_prob = predict_probabilities(lstm, x_sequences[sequence_test_mask])

        # The masks cover every validation/test target row because the sequence length is shorter than the train split.
        validation_predictions["prob_lstm"] = np.nan
        test_predictions["prob_lstm"] = np.nan
        validation_sequence_indices = sequence_target_indices[sequence_validation_mask]
        test_sequence_indices = sequence_target_indices[sequence_test_mask]
        validation_probability_map = dict(zip(validation_sequence_indices, lstm_val_prob))
        test_probability_map = dict(zip(test_sequence_indices, lstm_test_prob))
        validation_predictions["prob_lstm"] = validation_predictions["row_index"].map(
            validation_probability_map
        )
        test_predictions["prob_lstm"] = test_predictions["row_index"].map(
            test_probability_map
        )
        model_thresholds["lstm"] = lstm_threshold
        model_objects["lstm"] = lstm

        for split_name, y_true, probabilities in (
            ("validation", y_sequences[sequence_validation_mask].astype(int), lstm_val_prob),
            ("test", y_sequences[sequence_test_mask].astype(int), lstm_test_prob),
        ):
            metric_row = classification_metrics(y_true, probabilities, lstm_threshold)
            metric_row.update(
                {
                    "horizon_days": horizon,
                    "horizon_name": horizon_name,
                    "model": "lstm",
                    "split": split_name,
                    "best_epoch": lstm_history.best_epoch,
                }
            )
            all_result_rows.append(metric_row)
            if split_name == "validation":
                model_validation_auc["lstm"] = _safe_roc_auc(metric_row)

        torch.save(
            {
                "state_dict": lstm.state_dict(),
                "input_size": len(feature_columns),
                "sequence_length": CONFIG.sequence_length,
            },
            ARTIFACTS_DIR / "models" / f"{horizon_name}_lstm.pt",
        )

        validation_predictions.to_csv(
            ARTIFACTS_DIR / "predictions" / f"validation_predictions_{horizon_name}.csv",
            index=False,
        )
        test_predictions.to_csv(
            ARTIFACTS_DIR / "predictions" / f"test_predictions_{horizon_name}.csv",
            index=False,
        )

        best_model_name = max(model_validation_auc, key=model_validation_auc.get)
        best_threshold = model_thresholds[best_model_name]
        probability_column = f"prob_{best_model_name}"
        backtest, backtest_summary = create_non_overlapping_backtest(
            test_predictions,
            probability_column,
            best_threshold,
            horizon,
        )
        backtest.to_csv(
            ARTIFACTS_DIR / "backtests" / f"backtest_{horizon_name}.csv",
            index=False,
        )
        _plot_backtest(backtest, horizon, best_model_name)

        backtest_summary.update(
            {
                "horizon_days": horizon,
                "horizon_name": horizon_name,
                "model": best_model_name,
                "threshold": best_threshold,
            }
        )
        backtest_rows.append(backtest_summary)

        # Refit the validation-selected model on all labeled rows for deployment.
        # Evaluation metrics above remain based on the requested repeating 10-week split.
        latest_raw_x = latest_feature_df[feature_columns].to_numpy(dtype=np.float64)
        production_prefix = ARTIFACTS_DIR / "models" / f"production_{horizon_name}_{best_model_name}"

        if best_model_name in {"logistic_regression", "random_forest", "xgboost"}:
            production_model = clone(sklearn_models()[best_model_name])
            production_model.fit(x, y)
            production_path = production_prefix.with_suffix(".joblib")
            joblib.dump(production_model, production_path)
            latest_probability = float(production_model.predict_proba(latest_raw_x[-1:])[:, 1][0])
            model_type = "sklearn"
            production_scaler_path = None
        elif best_model_name == "mlp":
            production_scaler = StandardScaler().fit(x)
            production_x = production_scaler.transform(x).astype(np.float32)
            production_model = train_fixed_epochs(
                MLPClassifierNet(input_size=len(feature_columns)),
                production_x,
                y,
                epochs=mlp_history.best_epoch,
            )
            production_path = production_prefix.with_suffix(".pt")
            production_scaler_path = ARTIFACTS_DIR / "models" / f"production_{horizon_name}_scaler.joblib"
            joblib.dump(production_scaler, production_scaler_path)
            torch.save(
                {"state_dict": production_model.state_dict(), "input_size": len(feature_columns)},
                production_path,
            )
            latest_x = production_scaler.transform(latest_raw_x[-1:]).astype(np.float32)
            latest_probability = float(predict_probabilities(production_model, latest_x)[0])
            model_type = "pytorch_mlp"
        else:
            production_scaler = StandardScaler().fit(x)
            production_x = production_scaler.transform(x).astype(np.float32)
            production_sequences, production_labels, _ = create_sequences(
                production_x, y, CONFIG.sequence_length
            )
            production_model = train_fixed_epochs(
                LSTMClassifierNet(input_size=len(feature_columns)),
                production_sequences,
                production_labels,
                epochs=lstm_history.best_epoch,
                learning_rate=8e-4,
            )
            production_path = production_prefix.with_suffix(".pt")
            production_scaler_path = ARTIFACTS_DIR / "models" / f"production_{horizon_name}_scaler.joblib"
            joblib.dump(production_scaler, production_scaler_path)
            torch.save(
                {
                    "state_dict": production_model.state_dict(),
                    "input_size": len(feature_columns),
                    "sequence_length": CONFIG.sequence_length,
                },
                production_path,
            )
            latest_sequence = production_scaler.transform(
                latest_raw_x[-CONFIG.sequence_length :]
            ).astype(np.float32)[None, :, :]
            latest_probability = float(predict_probabilities(production_model, latest_sequence)[0])
            model_type = "pytorch_lstm"

        horizon_metadata[horizon_name] = {
            "days": horizon,
            "best_model": best_model_name,
            "best_model_type": model_type,
            "validation_roc_auc": model_validation_auc[best_model_name],
            "threshold": best_threshold,
            "latest_feature_date": latest_feature_df.iloc[-1]["date"].isoformat(),
            "latest_probability_up": latest_probability,
            "latest_direction": "UP" if latest_probability >= best_threshold else "DOWN",
            "production_model_path": str(production_path.relative_to(Path(__file__).resolve().parents[2])),
            "production_scaler_path": (
                str(production_scaler_path.relative_to(Path(__file__).resolve().parents[2]))
                if production_scaler_path is not None
                else None
            ),
            "all_validation_roc_auc": model_validation_auc,
        }

    results = pd.DataFrame(all_result_rows)
    results.to_csv(ARTIFACTS_DIR / "results" / "model_results.csv", index=False)
    backtest_results = pd.DataFrame(backtest_rows)
    backtest_results.to_csv(ARTIFACTS_DIR / "results" / "backtest_summary.csv", index=False)
    _plot_model_comparison(results)

    metadata = {
        "project": "Bitcoin direction prediction",
        "framework": "PyTorch",
        "interface": "Streamlit dashboard and Jupyter notebook",
        "data_path": str(Path(data_path).relative_to(Path(data_path).parents[2]))
        if len(Path(data_path).parents) >= 3
        else str(data_path),
        "targets": {
            "next_day": "close[t+1] > close[t]",
            "next_5_days": "close[t+5] > close[t]",
            "next_week": "close[t+7] > close[t]",
        },
        "feature_columns": feature_columns,
        "sequence_length": CONFIG.sequence_length,
        "split": split_info,
        "horizons": horizon_metadata,
        "limitations": [
            "The repeating week split mixes earlier and later market periods; it is an interleaved generalization test, not a strict future holdout.",
            "Historical performance does not imply future profitability.",
            "The dataset contains only Binance market variables and no news, macroeconomic, or on-chain data.",
            "The backtest uses non-overlapping horizon windows and simplified transaction costs.",
            "Predictions are for an academic portfolio project, not financial advice.",
        ],
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata
