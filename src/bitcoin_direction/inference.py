from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from .config import ARTIFACTS_DIR, CONFIG, PROJECT_ROOT
from .features import create_feature_table, load_raw_data
from .torch_models import LSTMClassifierNet, MLPClassifierNet, predict_probabilities


def predict_latest(csv_path: str | Path) -> pd.DataFrame:
    """Generate the latest UP/DOWN probability for each configured horizon."""
    metadata_path = ARTIFACTS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Run `python train.py` before requesting predictions.")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    raw = load_raw_data(str(csv_path))
    featured, generated_features = create_feature_table(raw, drop_unlabeled=False)
    feature_columns = metadata["feature_columns"]
    if generated_features != feature_columns:
        raise ValueError("Uploaded data produced a feature schema different from the trained model.")

    latest_raw_x = featured[feature_columns].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    for horizon_name, horizon_info in metadata["horizons"].items():
        model_type = horizon_info["best_model_type"]
        model_path = PROJECT_ROOT / horizon_info["production_model_path"]
        scaler_path_value = horizon_info.get("production_scaler_path")
        threshold = float(horizon_info["threshold"])

        if model_type == "sklearn":
            model = joblib.load(model_path)
            probability = float(model.predict_proba(latest_raw_x[-1:])[:, 1][0])
        else:
            if not scaler_path_value:
                raise ValueError(f"Missing scaler for {horizon_name}.")
            scaler = joblib.load(PROJECT_ROOT / scaler_path_value)
            checkpoint = torch.load(model_path, map_location="cpu")

            if model_type == "pytorch_mlp":
                model = MLPClassifierNet(input_size=int(checkpoint["input_size"]))
                model.load_state_dict(checkpoint["state_dict"])
                model_input = scaler.transform(latest_raw_x[-1:]).astype("float32")
            elif model_type == "pytorch_lstm":
                sequence_length = int(checkpoint.get("sequence_length", CONFIG.sequence_length))
                if len(latest_raw_x) < sequence_length:
                    raise ValueError(
                        f"At least {sequence_length} engineered rows are required for the LSTM."
                    )
                model = LSTMClassifierNet(input_size=int(checkpoint["input_size"]))
                model.load_state_dict(checkpoint["state_dict"])
                model_input = scaler.transform(
                    latest_raw_x[-sequence_length:]
                ).astype("float32")[None, :, :]
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            probability = float(predict_probabilities(model, model_input)[0])

        rows.append(
            {
                "horizon": horizon_name,
                "days": int(horizon_info["days"]),
                "data_date": featured.iloc[-1]["date"].strftime("%Y-%m-%d"),
                "model": horizon_info["best_model"],
                "probability_up": probability,
                "threshold": threshold,
                "prediction": "UP" if probability >= threshold else "DOWN",
            }
        )

    return pd.DataFrame(rows).sort_values("days").reset_index(drop=True)
