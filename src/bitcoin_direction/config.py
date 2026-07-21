from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "btc_usdt_1d_2018_2026.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class ProjectConfig:
    horizons: tuple[int, ...] = (1, 5, 7)
    horizon_names: dict[int, str] = field(
        default_factory=lambda: {1: "next_day", 5: "next_5_days", 7: "next_week"}
    )
    split_cycle_length_weeks: int = 10
    train_cycle_weeks: tuple[int, ...] = (1, 2, 4, 7, 9, 10)
    validation_cycle_weeks: tuple[int, ...] = (3, 6)
    test_cycle_weeks: tuple[int, ...] = (5, 8)
    sequence_length: int = 30
    random_state: int = 42
    transaction_fee_per_side: float = 0.001
    max_epochs: int = 60
    patience: int = 8
    batch_size: int = 64


CONFIG = ProjectConfig()
