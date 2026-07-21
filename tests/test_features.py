from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bitcoin_direction.features import (
    create_feature_table,
    load_raw_data,
    repeating_week_split_assignments,
    repeating_week_split_indices,
)


def test_feature_table_has_all_targets() -> None:
    raw = load_raw_data()
    featured, feature_columns = create_feature_table(raw)
    assert len(featured) > 2000
    assert len(feature_columns) >= 40
    assert featured[feature_columns].replace([np.inf, -np.inf], np.nan).notna().all().all()
    for horizon in (1, 5, 7):
        assert set(featured[f"target_{horizon}d"].unique()).issubset({0, 1})


def test_repeating_week_split_matches_requested_cycle() -> None:
    dates = pd.Series(pd.date_range("2024-01-01", periods=140, freq="D", tz="UTC"))
    assignments = repeating_week_split_assignments(dates, anchor_date="2024-01-01")

    expected = {
        1: "train",
        2: "train",
        3: "validation",
        4: "train",
        5: "test",
        6: "validation",
        7: "train",
        8: "test",
        9: "train",
        10: "train",
    }
    for cycle_week, split_name in expected.items():
        observed = set(assignments.loc[assignments["cycle_week"] == cycle_week, "split"])
        assert observed == {split_name}

    train_idx, validation_idx, test_idx, _ = repeating_week_split_indices(
        dates, anchor_date="2024-01-01"
    )
    assert len(train_idx) == 84
    assert len(validation_idx) == 28
    assert len(test_idx) == 28
