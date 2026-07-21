from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitcoin_direction.training import train_project


if __name__ == "__main__":
    metadata = train_project()
    print("\nTraining completed.\n")
    print(json.dumps(metadata["horizons"], indent=2))
