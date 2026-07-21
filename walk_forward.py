from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bitcoin_direction.walk_forward import run_walk_forward


if __name__ == "__main__":
    result = run_walk_forward()
    print(result.to_string(index=False))
