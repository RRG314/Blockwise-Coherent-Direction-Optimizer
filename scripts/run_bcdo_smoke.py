from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from optimizer_research import bcdo_default_config, load_yaml_config, run_bcdo_smoke  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    config = bcdo_default_config()
    if args.config:
        config.update(load_yaml_config(args.config))
    config.update(
        {
            "output_dir": config.get("output_dir", str(ROOT / "reports" / "bcdo_mainline")),
            "device": config.get("device", "cpu"),
            "smoke_seeds": config.get("smoke_seeds", [11]),
            "smoke_epoch_scale": config.get("smoke_epoch_scale", 0.35),
            "smoke_tasks": config.get("smoke_tasks", ["oscillatory_valley", "breast_cancer_mlp"]),
            "smoke_optimizers": config.get("smoke_optimizers", ["blockwise_consensus_direction_optimizer", "bcdo_structured_core", "adamw", "rmsprop"]),
        }
    )
    run_bcdo_smoke(config)
