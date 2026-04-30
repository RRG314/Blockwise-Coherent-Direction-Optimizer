from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import export_block_direction_v42_report  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "reports" / "block_direction_v42"))
    args = parser.parse_args()
    summary = export_block_direction_v42_report(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
