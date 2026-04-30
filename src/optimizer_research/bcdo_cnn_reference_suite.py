from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .benchmarking import _train_single_run, run_benchmark_suite, run_smoke_suite, run_tuning_suite
from .bcdo_literature import LITERATURE_ROWS
from .config import ensure_output_dir
from .reporting import (
    _load_trace_frames,
    _markdown_table,
    _plot_heatmap,
    _plot_metric,
    aggregate_results,
    best_by_task,
    compute_meaningful_wins,
)
from optimizers.optimizer_utils import resolve_device


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "bcdo_cnn_reference"

FOCUS_OPTIMIZERS = [
    "bcdo_cnn_reference",
    "blockwise_consensus_direction_optimizer",
    "magneto_hamiltonian_adam",
    "real_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "topological_adam",
]

BASELINE_COMPARISONS = [
    "blockwise_consensus_direction_optimizer",
    "magneto_hamiltonian_adam",
    "real_hamiltonian_adam",
    "adamw",
    "rmsprop",
    "sgd_momentum",
    "topological_adam",
]


def bcdo_cnn_reference_default_config() -> dict[str, Any]:
    return {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "device": "cpu",
        "seeds": [11, 29, 47],
        "search_budget": 2,
        "search_seed": 2201,
        "optimizers": FOCUS_OPTIMIZERS,
        "tuning_tasks": [
            "breast_cancer_mlp",
            "digits_mlp",
            "digits_cnn",
            "oscillatory_valley",
            "saddle_objective",
        ],
        "benchmark_tasks": [
            "breast_cancer_mlp",
            "digits_mlp",
            "digits_cnn",
            "wine_mlp",
            "oscillatory_valley",
            "saddle_objective",
            "plateau_escape_objective",
        ],
        "ablation_tasks": [
            "digits_cnn",
            "breast_cancer_mlp",
            "oscillatory_valley",
            "saddle_objective",
        ],
        "smoke_tasks": ["digits_cnn", "oscillatory_valley"],
        "smoke_optimizers": ["bcdo_cnn_reference", "blockwise_consensus_direction_optimizer", "adamw", "rmsprop"],
        "smoke_seeds": [11],
        "smoke_epoch_scale": 0.25,
        "tuning_epoch_scale": 0.45,
        "benchmark_epoch_scale": 0.75,
        "ablation_epoch_scale": 0.40,
        "use_tuning_results": True,
    }


def _prepare_docs(output_dir: Path) -> None:
    write_bcdo_cnn_reference_current_state(output_dir)
    write_bcdo_cnn_reference_literature_scan(output_dir)
    write_bcdo_cnn_reference_math_definition(output_dir)


def write_bcdo_cnn_reference_current_state(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BCDO CNN Reference Current State",
        "",
        "CNN reference is a separate branch built on top of the BCDO mainline and the earlier conv-safe scaling study.",
        "",
        "## Why CNN reference exists",
        "",
        "- BCDO mainline was the stronger main block branch.",
        "- Earlier conv-focused testing showed that conv-safe scaling helped, but filter-consensus as a default direction rule did not.",
        "- CNN reference keeps the helpful conv math and moves conv structure into trust scoring instead of adding another default candidate.",
        "",
        "## What CNN reference changes",
        "",
        "- Keeps the BCDO mainline candidate set unchanged.",
        "- Keeps conv-safe scaling.",
        "- Computes conv structure support from filter/channel/spatial/bank consistency.",
        "- Uses that support to boost stable-consensus and trusted-direction trust only on conv tensors.",
        "- Slightly relaxes fallback on coherent conv filters so the block selector can engage.",
        "",
    ]
    (output_path / "current_state.md").write_text("\n".join(lines), encoding="utf-8")


def write_bcdo_cnn_reference_literature_scan(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(LITERATURE_ROWS)
    frame.to_csv(output_path / "literature_matrix.csv", index=False)
    lines = [
        "# BCDO CNN Reference Literature Scan",
        "",
        "CNN reference keeps the same literature pressure as the block branch overall: it only matters if structure-aware blockwise direction choice beats simpler gradient transforms in the places where structure actually matters.",
        "",
        _markdown_table(
            frame[
                [
                    "family",
                    "representative_method",
                    "direction_or_transform",
                    "scope",
                    "difference_from_bcdo_reference",
                    "required_baseline",
                ]
            ]
        ),
        "",
        "## CNN reference-specific interpretation",
        "",
        "- This branch is still different from AdamW/RMSProp because the direction is selected blockwise rather than transformed by per-parameter moments.",
        "- Compared with Muon/Shampoo/K-FAC, the new idea is lightweight conv structure as a trust signal rather than a matrix preconditioner.",
        "- Compared with the earlier conv-focused study, the conv structure signal no longer claims to generate a better direction directly; it only influences trust and fallback.",
        "",
    ]
    for row in LITERATURE_ROWS:
        lines.append(f"- [{row['source_title']}]({row['source_url']})")
    lines.append("")
    (output_path / "literature_scan.md").write_text("\n".join(lines), encoding="utf-8")
    return frame


def write_bcdo_cnn_reference_math_definition(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BCDO CNN Reference Mathematical Definition",
        "",
        "CNN reference keeps the BCDO mainline candidate set",
        "",
        "`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`",
        "",
        "and changes only the conv-specific trust and step rules.",
        "",
        "For convolutional tensors `G in R^{O x I x S}` with filters `G_o`, define a conv-structure support score",
        "",
        "`R_o = 0.40 * align(G_o, channel_profile_o) + 0.35 * align(G_o, spatial_profile_o) + 0.25 * align(G_o, bank_profile)`",
        "",
        "where each `align` term is the positive cosine agreement between the normalized filter direction and the corresponding broadcast profile.",
        "",
        "The trust score becomes",
        "",
        "`T_i(d) = T_i^(v4)(d) + b_c * R_i * C_i(d) + b_m * R_i * M_i(d)`",
        "",
        "for the `stable_consensus` and `trusted_direction` candidates on conv tensors, and unchanged elsewhere.",
        "",
        "Fallback on conv tensors becomes",
        "",
        "`tau_i^(conv) = max(0, tau - rho * R_i)`",
        "",
        "so coherent conv filters are less likely to fall back immediately to the raw gradient.",
        "",
        "The conv-safe step scaling from the earlier conv-focused study remains:",
        "",
        "- lower conv update cap",
        "- stronger energy-power term when conv support is weak",
        "- bounded conv step multiplier",
        "",
    ]
    (output_path / "math_definition.md").write_text("\n".join(lines), encoding="utf-8")


def run_bcdo_cnn_reference_smoke(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_smoke_suite(config)


def run_bcdo_cnn_reference_tuning(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    return run_tuning_suite(config)


def _write_win_flags(output_path: Path, aggregated: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for baseline_name in BASELINE_COMPARISONS:
        if baseline_name not in set(aggregated["optimizer"]):
            continue
        wins = compute_meaningful_wins(aggregated, "bcdo_cnn_reference", baseline_name)
        if not wins.empty:
            frames.append(wins)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["task", "optimizer", "baseline", "win", "two_x", "rationale"])
    frame.to_csv(output_path / "win_flags.csv", index=False)
    return frame


def run_bcdo_cnn_reference_benchmarks(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    frame = run_benchmark_suite(config)
    aggregated = aggregate_results(frame)
    best_by_task(aggregated).to_csv(output_dir / "best_by_task.csv", index=False)
    _write_win_flags(output_dir, aggregated)
    return frame


def run_bcdo_cnn_reference_ablation(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = ensure_output_dir(config)
    _prepare_docs(output_dir)
    device = resolve_device(str(config.get("device", "cpu")))
    seeds = list(config.get("seeds", [11, 29, 47]))
    task_names = list(config.get("ablation_tasks", ["digits_cnn", "breast_cancer_mlp", "oscillatory_valley"]))
    variants = [
        {"variant_name": "v42_full", "optimizer_name": "bcdo_cnn_reference", "overrides": {}},
        {"variant_name": "no_conv_structure_bonus", "optimizer_name": "bcdo_cnn_reference", "overrides": {"conv_consensus_bonus": 0.0, "conv_memory_bonus": 0.0}},
        {"variant_name": "no_conv_fallback_relaxation", "optimizer_name": "bcdo_cnn_reference", "overrides": {"conv_fallback_relaxation": 0.0}},
        {"variant_name": "no_conv_safe_scaling", "optimizer_name": "bcdo_cnn_reference", "overrides": {"conv_max_update_ratio": 0.16, "conv_energy_power_bonus": 0.0, "conv_step_floor": 1.0}},
                {"variant_name": "v4_fast_baseline", "optimizer_name": "blockwise_consensus_direction_optimizer", "overrides": {}},
        {"variant_name": "adamw_baseline", "optimizer_name": "adamw", "overrides": {}},
        {"variant_name": "rmsprop_baseline", "optimizer_name": "rmsprop", "overrides": {}},
    ]

    rows: list[dict[str, Any]] = []
    for task_name in task_names:
        for variant in variants:
            for seed in seeds:
                row = _train_single_run(
                    suite_name="bcdo_cnn_reference_ablation",
                    task_name=task_name,
                    optimizer_name=str(variant["optimizer_name"]),
                    hyperparameters=dict(variant["overrides"]),
                    seed=seed,
                    device=device,
                    output_dir=output_dir,
                    save_trace=False,
                    epoch_scale=float(config.get("ablation_epoch_scale", 0.40)),
                )
                row["variant_name"] = variant["variant_name"]
                row["reference_optimizer"] = variant["optimizer_name"]
                row["variant_overrides"] = json.dumps(variant["overrides"], sort_keys=True, default=str)
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_results.csv", index=False)
    return frame


def _best_row_for_optimizer(aggregated: pd.DataFrame, optimizer_name: str) -> pd.Series | None:
    frame = aggregated[aggregated["optimizer"] == optimizer_name]
    if frame.empty:
        return None
    if frame["mean_best_val_accuracy"].notna().any():
        return frame.sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]
    return frame.sort_values(["mean_best_val_loss", "mean_runtime_seconds"], ascending=[True, True]).iloc[0]


def _variant_delta(frame: pd.DataFrame, full_name: str, compare_name: str) -> float | None:
    summary = frame.groupby("variant_name", as_index=False)["selection_score"].mean()
    if full_name not in set(summary["variant_name"]) or compare_name not in set(summary["variant_name"]):
        return None
    full_score = float(summary.loc[summary["variant_name"] == full_name, "selection_score"].iloc[0])
    compare_score = float(summary.loc[summary["variant_name"] == compare_name, "selection_score"].iloc[0])
    return full_score - compare_score


def _format_optional_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _plot_bar(frame: pd.DataFrame, output_path: Path, x_col: str, y_col: str, title: str, rotation: int = 35) -> None:
    if frame.empty or x_col not in frame.columns or y_col not in frame.columns:
        return
    ordered = frame.sort_values(y_col, ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(ordered[x_col], ordered[y_col])
    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def export_bcdo_cnn_reference_report(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_path = Path(output_dir)
    _prepare_docs(output_path)
    benchmark_frame = pd.read_csv(output_path / "benchmark_results.csv")
    tuning_frame = pd.read_csv(output_path / "tuning_results.csv")
    ablation_frame = pd.read_csv(output_path / "ablation_results.csv")

    aggregated = aggregate_results(benchmark_frame)
    best_frame = best_by_task(aggregated)
    best_frame.to_csv(output_path / "best_by_task.csv", index=False)
    win_flags = _write_win_flags(output_path, aggregated)

    v42_row = _best_row_for_optimizer(aggregated, "bcdo_cnn_reference")
    v4_row = _best_row_for_optimizer(aggregated, "blockwise_consensus_direction_optimizer")
    strongest_baseline = aggregated[
        aggregated["optimizer"].isin(["adamw", "rmsprop", "sgd_momentum", "topological_adam"])
    ].sort_values(["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]).iloc[0]

    win_map = {
        baseline: compute_meaningful_wins(aggregated, "bcdo_cnn_reference", baseline)
        for baseline in BASELINE_COMPARISONS
        if baseline in set(aggregated["optimizer"])
    }
    ablation_summary = (
        ablation_frame.groupby("variant_name", as_index=False)["selection_score"]
        .mean()
        .rename(columns={"selection_score": "mean_selection_score"})
        .sort_values("mean_selection_score", ascending=False)
    )
    structure_delta = _variant_delta(ablation_frame, "v42_full", "no_conv_structure_bonus")
    fallback_delta = _variant_delta(ablation_frame, "v42_full", "no_conv_fallback_relaxation")
    scaling_delta = _variant_delta(ablation_frame, "v42_full", "no_conv_safe_scaling")

    figure_dir = output_path / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_frame = _load_trace_frames(benchmark_frame) if "trace_path" in benchmark_frame.columns else pd.DataFrame()
    comparison_optimizers = [
        "bcdo_cnn_reference",
        "blockwise_consensus_direction_optimizer",        "magneto_hamiltonian_adam",
        "adamw",
        "rmsprop",
        "sgd_momentum",
    ]
    _plot_metric(
        trace_frame,
        output_path=figure_dir / "loss_curves.png",
        title="Validation loss curves",
        metric="val_loss",
        tasks=["breast_cancer_mlp", "digits_cnn", "oscillatory_valley"],
        optimizers=comparison_optimizers,
        event="val",
    )
    _plot_heatmap(aggregated, figure_dir / "win_loss_heatmap.png")
    _plot_bar(ablation_summary, figure_dir / "ablation_chart.png", "variant_name", "mean_selection_score", "BCDO CNN Reference ablation", 45)

    lines = [
        "# BCDO CNN Reference Final Report",
        "",
        "## 1. What CNN reference is",
        "",
        "- CNN reference is a conv-aware refinement of BCDO mainline that uses conv structure as trust, not as a default new direction.",
        "- It keeps the BCDO mainline candidate set unchanged and uses conv-safe scaling plus conv-aware trust/fallback rules.",
        "",
        "## 2. Best rows",
        "",
        f"- Best CNN reference row: `{v42_row['task']}` with mean best val loss `{float(v42_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v42_row['mean_best_val_accuracy']):.6f}`." if v42_row is not None else "- Best CNN reference row: unavailable.",
        f"- Best BCDO mainline row on the same suite: `{v4_row['task']}` with mean best val loss `{float(v4_row['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(v4_row['mean_best_val_accuracy']):.6f}`." if v4_row is not None else "- Best BCDO mainline row: unavailable.",
        f"- Strongest baseline row: `{strongest_baseline['optimizer']}` on `{strongest_baseline['task']}` with mean best val loss `{float(strongest_baseline['mean_best_val_loss']):.6f}` and mean best val accuracy `{float(strongest_baseline['mean_best_val_accuracy']):.6f}`.",
        "",
        "## 3. Competitive summary",
        "",
        f"- CNN reference wins vs BCDO mainline: `{int(win_map.get('blockwise_consensus_direction_optimizer', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- CNN reference wins vs AdamW: `{int(win_map.get('adamw', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- CNN reference wins vs RMSProp: `{int(win_map.get('rmsprop', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- CNN reference wins vs SGD momentum: `{int(win_map.get('sgd_momentum', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        f"- CNN reference wins vs the internal coherent momentum comparator family: `{int(win_map.get('magneto_hamiltonian_adam', pd.DataFrame()).get('win', pd.Series(dtype=int)).sum())}` meaningful wins.",
        "",
        "## 4. Ablation findings",
        "",
        f"- Conv structure bonus delta vs removal: `{_format_optional_delta(structure_delta)}`.",
        f"- Conv fallback relaxation delta vs removal: `{_format_optional_delta(fallback_delta)}`.",
        f"- Conv safe scaling delta vs removal: `{_format_optional_delta(scaling_delta)}`.",
        f"- Best ablation row overall: `{str(ablation_summary.iloc[0]['variant_name'])}`." if not ablation_summary.empty else "- Best ablation row overall: unavailable.",
        "",
        "## 5. Honest conclusion",
        "",
        "- CNN reference only succeeds if it improves standard/CNN behavior without giving back BCDO mainline's stress-task wins.",
        "- If RMSProp still wins CNN tasks cleanly, CNN reference remains a specialist rather than a broadly usable default.",
        "",
    ]
    (output_path / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "best_v42_task": None if v42_row is None else str(v42_row["task"]),
        "best_v42_loss": None if v42_row is None else float(v42_row["mean_best_val_loss"]),
        "best_v42_accuracy": None if v42_row is None else float(v42_row["mean_best_val_accuracy"]),
        "benchmark_rows": int(len(benchmark_frame)),
        "tuning_rows": int(len(tuning_frame)),
        "ablation_rows": int(len(ablation_frame)),
        "win_flag_rows": int(len(win_flags)),
    }
