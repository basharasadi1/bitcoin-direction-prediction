from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitcoin_direction.config import DATA_PATH
from bitcoin_direction.inference import predict_latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict BTC direction from a compatible daily CSV.")
    parser.add_argument("csv", nargs="?", default=str(DATA_PATH), help="Path to a Binance-format daily CSV")
    args = parser.parse_args()
    predictions = predict_latest(args.csv)
    print(predictions.to_string(index=False, formatters={"probability_up": "{:.3f}".format, "threshold": "{:.3f}".format}))


if __name__ == "__main__":
    main()
