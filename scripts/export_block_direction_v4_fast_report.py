from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import export_block_direction_v4_fast_report  # noqa: E402


if __name__ == "__main__":
    export_block_direction_v4_fast_report(ROOT / "reports" / "bcdo_mainline")
