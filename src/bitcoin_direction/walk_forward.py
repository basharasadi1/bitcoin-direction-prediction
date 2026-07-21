from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import ARTIFACTS_DIR, CONFIG, DATA_PATH
from .features import create_feature_table, load_raw_data
from .metrics import classification_metrics
from .training import ensure_directories


def _fast_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=CONFIG.random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=60, max_depth=6, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced_subsample", random_state=CONFIG.random_state, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=40, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.80, min_child_weight=5, reg_lambda=2.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=CONFIG.random_state, n_jobs=2, tree_method="hist",
        ),
    }


def run_walk_forward(data_path: str | Path = DATA_PATH, n_splits: int = 3) -> pd.DataFrame:
    """Expanding-window evaluation for the classical models."""
    ensure_directories()
    raw = load_raw_data(str(data_path))
    df, feature_columns = create_feature_table(raw, drop_unlabeled=True)
    x = df[feature_columns].to_numpy(dtype=float)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, object]] = []

    for horizon in CONFIG.horizons:
        y = df[f"target_{horizon}d"].to_numpy(dtype=int)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x), start=1):
            for model_name, base_model in _fast_models().items():
                model = clone(base_model)
                model.fit(x[train_idx], y[train_idx])
                probabilities = model.predict_proba(x[test_idx])[:, 1]
                metrics = classification_metrics(y[test_idx], probabilities, threshold=0.5)
                metrics.update(
                    {
                        "horizon_days": horizon,
                        "horizon_name": CONFIG.horizon_names[horizon],
                        "model": model_name,
                        "fold": fold,
                        "train_rows": len(train_idx),
                        "test_rows": len(test_idx),
                        "test_start": df.iloc[test_idx[0]]["date"].strftime("%Y-%m-%d"),
                        "test_end": df.iloc[test_idx[-1]]["date"].strftime("%Y-%m-%d"),
                    }
                )
                rows.append(metrics)

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(ARTIFACTS_DIR / "results" / "walk_forward_folds.csv", index=False)

    summary = (
        fold_results.groupby(["horizon_days", "horizon_name", "model"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
        )
    )
    summary.to_csv(ARTIFACTS_DIR / "results" / "walk_forward_summary.csv", index=False)

    pivot = summary.pivot(index="model", columns="horizon_name", values="mean_roc_auc")
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Walk-Forward Mean ROC-AUC")
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "figures" / "walk_forward_roc_auc.png", dpi=160)
    plt.close()

    return summary
