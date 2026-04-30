from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import bcdo_default_config, load_yaml_config, run_bcdo_tuning  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "bcdo_tuning.yaml"))
    args = parser.parse_args()

    config = bcdo_default_config()
    config.update(load_yaml_config(args.config))
    run_bcdo_tuning(config)
