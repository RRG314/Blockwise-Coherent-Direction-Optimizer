from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_DIR = ROOT / "reports" / "accepted_bcdo"


def test_accepted_bcdo_outputs_exist_and_match_schema() -> None:
    benchmark_path = ACCEPTED_DIR / "benchmark_results.csv"
    best_path = ACCEPTED_DIR / "best_by_task.csv"
    win_flags_path = ACCEPTED_DIR / "win_flags.csv"
    final_report_path = ACCEPTED_DIR / "final_report.md"

    assert benchmark_path.exists()
    assert best_path.exists()
    assert win_flags_path.exists()
    assert final_report_path.exists()

    benchmark = pd.read_csv(benchmark_path)
    assert {
        "optimizer",
        "task",
        "suite",
        "best_val_loss",
        "best_val_accuracy",
        "selection_score",
        "runtime_per_step_ms",
        "optimizer_state_mb",
    }.issubset(benchmark.columns)

    accepted = benchmark[benchmark["optimizer"] == "blockwise_consensus_direction_optimizer"]
    aggregated = (
        accepted.groupby("task", as_index=False)[["best_val_loss", "best_val_accuracy"]]
        .mean()
        .rename(columns={"best_val_loss": "mean_best_val_loss", "best_val_accuracy": "mean_best_val_accuracy"})
    )
    breast = aggregated[aggregated["task"] == "breast_cancer_mlp"].iloc[0]
    digits_cnn = aggregated[aggregated["task"] == "digits_cnn"].iloc[0]
    assert abs(float(breast["mean_best_val_loss"]) - 0.06172556938448296) < 1e-6
    assert abs(float(breast["mean_best_val_accuracy"]) - 0.9856770833333334) < 1e-9
    assert abs(float(digits_cnn["mean_best_val_loss"]) - 0.8142097269495329) < 1e-6
    assert abs(float(digits_cnn["mean_best_val_accuracy"]) - 0.740293562412262) < 1e-9


def test_accepted_bcdo_win_flags_contain_expected_baselines() -> None:
    win_flags = pd.read_csv(ACCEPTED_DIR / "win_flags.csv")
    assert {"optimizer", "baseline", "win", "two_x", "task"}.issubset(win_flags.columns)
    baselines = set(win_flags["baseline"].unique())
    assert {"adamw", "rmsprop", "sgd_momentum", "magneto_hamiltonian_adam"}.issubset(baselines)
