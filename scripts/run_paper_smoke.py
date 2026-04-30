from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = [
    ROOT / "paper" / "bcdo_draft.md",
    ROOT / "paper" / "bcdo_claims_audit.md",
    ROOT / "paper" / "bcdo_results_summary.csv",
    ROOT / "paper" / "tables" / "accepted_bcdo_summary.md",
    ROOT / "paper" / "figures" / "accepted_best_val_loss_by_task.png",
]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    commands = [
        [sys.executable, "-m", "compileall", "src"],
        [sys.executable, "-m", "pytest", "tests/test_bcdo.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/test_blockwise_consensus_direction_optimizer.py", "tests/test_bcdo_benchmark_outputs.py", "-q"],
        [sys.executable, "examples/bcdo_minimal_mlp.py"],
        [sys.executable, "examples/bcdo_blockwise_direction_demo.py"],
        [sys.executable, "examples/bcdo_compare_against_adamw.py"],
        [sys.executable, "scripts/build_paper_artifacts.py"],
    ]
    for command in commands:
        _run(command)
    missing = [path for path in EXPECTED_FILES if not path.exists()]
    if missing:
        print("\nPaper smoke failed. Missing artifacts:")
        for path in missing:
            print(f"- {path}")
        return 1
    print("\nPaper smoke passed.")
    for path in EXPECTED_FILES:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
