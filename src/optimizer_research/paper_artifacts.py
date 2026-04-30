from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .reporting import _markdown_table, aggregate_results, compute_meaningful_wins


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
PAPER_DIR = ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

ACCEPTED_DIR = REPORTS_DIR / "accepted_bcdo"
MAINLINE_DIR = REPORTS_DIR / "bcdo_mainline"
REFERENCE_CNN_DIR = REPORTS_DIR / "reference_cnn_branch"
CNN_PROBE_DIR = REPORTS_DIR / "bcdo_cnn_probe"
PINN_PROBE_DIR = REPORTS_DIR / "bcdo_pinn_probe"

PUBLIC_OPTIMIZERS = [
    "blockwise_consensus_direction_optimizer",
    "bcdo_mainline_legacy",
    "bcdo_cnn_reference",
    "adamw",
    "rmsprop",
    "sgd_momentum",
]


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _ensure_dirs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _write_table(name: str, frame: pd.DataFrame, sources: list[Path], intro: str) -> None:
    table_frame = frame.copy()
    table_frame["source_csv"] = " | ".join(_relative(path) for path in sources)
    csv_path = TABLES_DIR / f"{name}.csv"
    md_path = TABLES_DIR / f"{name}.md"
    table_frame.to_csv(csv_path, index=False)
    lines = [
        f"# {name.replace('_', ' ').title()}",
        "",
        intro,
        "",
        "Source CSVs:",
    ]
    lines.extend(f"- `{_relative(path)}`" for path in sources)
    lines.extend(["", _markdown_table(table_frame)])
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_grouped_bar(
    frame: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    title: str,
    output_path: Path,
    ylabel: str,
) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index=category_col, columns=group_col, values=value_col)
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title=group_col, fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_simple_bar(frame: pd.DataFrame, label_col: str, value_col: str, title: str, output_path: Path, ylabel: str) -> None:
    if frame.empty:
        return
    ordered = frame.sort_values(value_col, ascending=False)
    plt.figure(figsize=(9, 4.5))
    plt.bar(ordered[label_col], ordered[value_col])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _summarize_wins(aggregated: pd.DataFrame, optimizer_name: str, baselines: list[str], benchmark_group: str, source_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    present = set(aggregated["optimizer"])
    if optimizer_name not in present:
        return pd.DataFrame(columns=["benchmark_group", "optimizer", "baseline", "meaningful_wins", "two_x_wins", "source_csv"])
    for baseline in baselines:
        if baseline not in present:
            continue
        wins = compute_meaningful_wins(aggregated, optimizer_name, baseline)
        rows.append(
            {
                "benchmark_group": benchmark_group,
                "optimizer": optimizer_name,
                "baseline": baseline,
                "meaningful_wins": int(wins["win"].sum()),
                "two_x_wins": int(wins["two_x"].sum()) if "two_x" in wins.columns else 0,
                "source_csv": _relative(source_csv),
            }
        )
    return pd.DataFrame(rows)


def _accepted_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    tasks = [
        "breast_cancer_mlp",
        "wine_mlp",
        "digits_mlp",
        "digits_cnn",
        "oscillatory_valley",
        "plateau_escape_objective",
        "saddle_objective",
        "low_rank_matrix_objective",
    ]
    return aggregated[
        aggregated["task"].isin(tasks) & aggregated["optimizer"].isin(PUBLIC_OPTIMIZERS)
    ][
        [
            "task",
            "optimizer",
            "task_family",
            "mean_best_val_loss",
            "mean_best_val_accuracy",
            "mean_runtime_per_step_ms",
            "mean_optimizer_state_mb",
        ]
    ].sort_values(["task", "optimizer"])


def _current_vs_accepted_table(accepted_agg: pd.DataFrame, runnable_agg: pd.DataFrame) -> pd.DataFrame:
    accepted = accepted_agg[accepted_agg["optimizer"] == "blockwise_consensus_direction_optimizer"][
        ["task", "mean_best_val_loss", "mean_best_val_accuracy", "mean_runtime_per_step_ms"]
    ].copy()
    accepted["line"] = "accepted_public_snapshot"
    runnable = runnable_agg[runnable_agg["optimizer"] == "blockwise_consensus_direction_optimizer"][
        ["task", "mean_best_val_loss", "mean_best_val_accuracy", "mean_runtime_per_step_ms"]
    ].copy()
    runnable["line"] = "current_runnable_output"
    return pd.concat([accepted, runnable], ignore_index=True).sort_values(["task", "line"])


def _runtime_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    return (
        aggregated.groupby("optimizer", as_index=False)[["mean_runtime_per_step_ms", "mean_optimizer_state_mb"]]
        .mean(numeric_only=True)
        .sort_values("mean_runtime_per_step_ms")
    )


def _reference_cnn_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated[
        aggregated["optimizer"].isin(
            [
                "blockwise_consensus_direction_optimizer",
                "bcdo_cnn_reference",
                "adamw",
                "rmsprop",
                "sgd_momentum",
            ]
        )
    ][
        [
            "task",
            "optimizer",
            "mean_best_val_loss",
            "mean_best_val_accuracy",
            "mean_runtime_per_step_ms",
        ]
    ].sort_values(["task", "optimizer"])


def _pinn_table(aggregated: pd.DataFrame) -> pd.DataFrame:
    return aggregated[
        aggregated["optimizer"].isin(
            [
                "blockwise_consensus_direction_optimizer",
                "adamw",
                "adamw_lbfgs_hybrid",
                "lbfgs",
                "rmsprop",
                "sgd_momentum",
            ]
        )
    ][
        [
            "task",
            "optimizer",
            "mean_best_val_loss",
            "mean_runtime_per_step_ms",
        ]
    ].sort_values(["task", "optimizer"])


def _ablation_table(frame: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in ["best_val_loss", "best_val_accuracy", "runtime_per_step_ms", "selection_score"] if col in frame.columns]
    grouped = frame.groupby("variant_name", as_index=False)[value_cols].mean(numeric_only=True)
    rename_map = {
        "best_val_loss": "mean_best_val_loss",
        "best_val_accuracy": "mean_best_val_accuracy",
        "runtime_per_step_ms": "mean_runtime_per_step_ms",
        "selection_score": "mean_selection_score",
    }
    grouped = grouped.rename(columns=rename_map)
    sort_col = "mean_selection_score" if "mean_selection_score" in grouped.columns else "mean_best_val_loss"
    ascending = False if sort_col == "mean_selection_score" else True
    return grouped.sort_values(sort_col, ascending=ascending)


def _write_claims_audit(
    accepted_win_summary: pd.DataFrame,
    cnn_probe_agg: pd.DataFrame,
    pinn_probe_agg: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> Path:
    wins_vs_adamw = accepted_win_summary.loc[accepted_win_summary["baseline"] == "adamw", "meaningful_wins"]
    wins_vs_rmsprop = accepted_win_summary.loc[accepted_win_summary["baseline"] == "rmsprop", "meaningful_wins"]
    wins_vs_sgdm = accepted_win_summary.loc[accepted_win_summary["baseline"] == "sgd_momentum", "meaningful_wins"]
    cnn_best = cnn_probe_agg[(cnn_probe_agg["task"] == "digits_cnn") & (cnn_probe_agg["optimizer"].isin(["blockwise_consensus_direction_optimizer", "adamw", "rmsprop", "sgd_momentum"]))][
        ["optimizer", "mean_best_val_accuracy"]
    ].sort_values("mean_best_val_accuracy", ascending=False)
    pinn_best = pinn_probe_agg[
        pinn_probe_agg["optimizer"].isin(["blockwise_consensus_direction_optimizer", "lbfgs", "adamw_lbfgs_hybrid"])
    ][["optimizer", "mean_best_val_loss"]].sort_values("mean_best_val_loss")
    runtime_lookup = runtime_table.set_index("optimizer")
    bcdo_runtime = runtime_lookup.loc["blockwise_consensus_direction_optimizer", "mean_runtime_per_step_ms"] if "blockwise_consensus_direction_optimizer" in runtime_lookup.index else float("nan")
    adamw_runtime = runtime_lookup.loc["adamw", "mean_runtime_per_step_ms"] if "adamw" in runtime_lookup.index else float("nan")
    lines = [
        "# BCDO Claims Audit",
        "",
        "This file is the paper-facing claim check for the public Blockwise Consensus Direction Optimizer line.",
        "",
        "## Supported claims",
        "",
        "The checked-in accepted snapshot supports presenting BCDO as a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, or structurally misleading at the block level. The accepted first-release line still shows meaningful wins over AdamW and real stress-task strength while keeping the design materially different from moment-only adaptive baselines.",
        "",
        f"In the accepted public snapshot BCDO shows `{int(wins_vs_adamw.iloc[0]) if not wins_vs_adamw.empty else 0}` meaningful wins over AdamW, `{int(wins_vs_rmsprop.iloc[0]) if not wins_vs_rmsprop.empty else 0}` over RMSProp, and `{int(wins_vs_sgdm.iloc[0]) if not wins_vs_sgdm.empty else 0}` over SGD with momentum. Those counts are enough to justify a narrow structured-specialist claim.",
        "",
        "## Claims the repository does not support",
        "",
        "The repository does not support a universal optimizer claim, a broad CNN claim, or a PINN claim. It also does not support claiming that every branch experiment should be promoted into the public line. The accepted snapshot stays separate for that reason.",
        "",
        "## Failure cases that should remain visible",
        "",
        "The CNN probe still shows the practical baselines ahead on the digits CNN slice, and the PINN probe still favors closure-based alternatives such as LBFGS or AdamW-to-LBFGS hybrids. Those are boundary conditions, not footnotes.",
        "",
        "## Unfair claims",
        "",
        "It would be unfair to claim that BCDO is a strong CNN optimizer, a PINN optimizer, or a general replacement for RMSProp, SGD with momentum, or AdamW. It would also be unfair to blur the accepted public snapshot together with later runnable or exploratory output folders.",
        "",
        "## Safe public wording",
        "",
        "BCDO is a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, or structurally misleading. The accepted snapshot shows real wins on stress-oriented and structure-sensitive slices, but it does not replace the strongest simple baselines as a general default optimizer.",
        "",
        "## Runtime context",
        "",
        f"The accepted public line currently averages `{_fmt(bcdo_runtime, 4)} ms` per step against `{_fmt(adamw_runtime, 4)} ms` for AdamW in the checked-in snapshot. That runtime gap is part of the honest public story.",
        "",
        "## Probe reminders",
        "",
        "The CNN and PINN probes remain visible because they show where the method is still weak. Hiding those reports would make the repository less credible, not more.",
        "",
        "## Sources",
        "",
        "- `reports/accepted_bcdo/benchmark_results.csv`",
        "- `reports/accepted_bcdo/win_flags.csv`",
        "- `reports/reference_cnn_branch/benchmark_results.csv`",
        "- `reports/bcdo_cnn_probe/benchmark_results.csv`",
        "- `reports/bcdo_pinn_probe/benchmark_results.csv`",
    ]
    output_path = PAPER_DIR / "bcdo_claims_audit.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _write_paper_draft(
    accepted_best_row: pd.Series,
    accepted_win_summary: pd.DataFrame,
    runtime_table: pd.DataFrame,
) -> Path:
    wins_vs_adamw = accepted_win_summary.loc[accepted_win_summary["baseline"] == "adamw", "meaningful_wins"]
    wins_vs_rmsprop = accepted_win_summary.loc[accepted_win_summary["baseline"] == "rmsprop", "meaningful_wins"]
    wins_vs_sgdm = accepted_win_summary.loc[accepted_win_summary["baseline"] == "sgd_momentum", "meaningful_wins"]
    wins_vs_magneto = accepted_win_summary.loc[accepted_win_summary["baseline"] == "magneto_hamiltonian_adam", "meaningful_wins"]
    runtime_lookup = runtime_table.set_index("optimizer")
    bcdo_runtime = runtime_lookup.loc["blockwise_consensus_direction_optimizer", "mean_runtime_per_step_ms"] if "blockwise_consensus_direction_optimizer" in runtime_lookup.index else float("nan")
    adamw_runtime = runtime_lookup.loc["adamw", "mean_runtime_per_step_ms"] if "adamw" in runtime_lookup.index else float("nan")
    text = f"""# Blockwise Consensus Direction Optimizer: Blockwise Direction Selection Under Unstable Local Geometry

## Abstract

This draft documents the current public claim supported by the repository: Blockwise Consensus Direction Optimizer is a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, structurally misleading, or too myopic to trust block-for-block. The method forms a small candidate set per parameter block, scores those candidates with structural trust signals, selects one winner, and then applies a bounded blockwise step. The checked-in accepted snapshot supports a narrow claim. BCDO shows meaningful wins over AdamW and several structure-sensitive or stress-oriented strengths, but it remains slower than simpler baselines and does not support broad claims on CNNs or PINNs.

## Introduction

Most first-order optimizers accept the current gradient direction and focus on smoothing, scaling, or preconditioning that direction. BCDO treats the problem differently. It asks whether the block should follow the raw gradient at all, or whether a consensus, memory, or matrix-structured candidate is more trustworthy for that block at that step.

This shift is the main reason the repository keeps BCDO separate from AdamW-style adaptive methods. The public story is not that BCDO has a slightly different adaptive denominator. The story is that it chooses between competing block directions before it scales the step.

## Related Work

The closest baseline families are SGD and momentum, RMSProp, Adam and AdamW, and structure-aware optimizers such as Muon, Shampoo, and K-FAC. BCDO shares the first-order training-loop footprint of the simpler baselines, but it uses block structure for direction choice rather than for tensor preconditioning. SAM and ASAM matter as stability comparisons, while PCGrad and CAGrad matter as conflict-aware references, but neither family uses the same blockwise candidate-selection rule.

## Method

Let `g_i` be the gradient restricted to block `i`. BCDO forms a small candidate set `D_i = {{d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}}`, scores each candidate with a trust function that combines descent alignment, memory coherence, quality memory, norm stability, consensus support, oscillation penalty, conflict penalty, and a practical cost term, and selects the winning direction with

`d_i* = argmax_d T_i(d)`.

After direction choice, the optimizer applies a bounded block-energy step

`alpha_i = lr * trust_i * ||g_i|| / (sqrt(E_i) + eps)^p`.

This keeps direction choice and magnitude control separate. That separation is the main algorithmic difference between BCDO and standard adaptive baselines.

## Experimental Setup

This draft uses only local checked-in artifacts:

- accepted public snapshot: `reports/accepted_bcdo/benchmark_results.csv`
- current runnable output path: `reports/bcdo_mainline/benchmark_results.csv`
- reference CNN branch: `reports/reference_cnn_branch/benchmark_results.csv`
- CNN probe: `reports/bcdo_cnn_probe/benchmark_results.csv`
- PINN probe: `reports/bcdo_pinn_probe/benchmark_results.csv`

The accepted public snapshot and the current runnable output path are kept separate on purpose. The accepted snapshot is the fixed first-release public evidence line. The runnable directory is the writable location used by local scripts.

## Results

The best accepted BCDO row is `{accepted_best_row['task']}` with mean best validation loss `{_fmt(accepted_best_row['mean_best_val_loss'], 6)}` and mean best validation accuracy `{_fmt(accepted_best_row['mean_best_val_accuracy'], 6)}`. The accepted snapshot shows `{int(wins_vs_adamw.iloc[0]) if not wins_vs_adamw.empty else 0}` meaningful wins over AdamW, `{int(wins_vs_rmsprop.iloc[0]) if not wins_vs_rmsprop.empty else 0}` over RMSProp, `{int(wins_vs_sgdm.iloc[0]) if not wins_vs_sgdm.empty else 0}` over SGD with momentum, and `{int(wins_vs_magneto.iloc[0]) if not wins_vs_magneto.empty else 0}` over the internal Coherent Momentum comparator.

These results support a narrow claim: blockwise candidate selection can be useful on structure-sensitive and stress-oriented slices. They do not support a claim that BCDO replaces the strongest simple baselines broadly.

## Ablations

The accepted ablation report is important because it shows that the public line is not just a random accumulation of branch logic. Block structure, matrix consensus, typed conv and dense profiles, cheap conv structure support, stable consensus, and trusted-direction memory remain the parts that survive the public line. Recoverability did not survive the accepted default path. The public method should therefore be described as the small candidate-set selector that remained after branch pruning, not as the union of every explored idea.

## Runtime and Memory Costs

The accepted public line currently averages `{_fmt(bcdo_runtime, 4)} ms` per step against `{_fmt(adamw_runtime, 4)} ms` for AdamW in the checked-in snapshot. That runtime cost narrows the set of workloads where the blockwise logic is worth using. The method only makes sense when blockwise directional reliability is a real bottleneck.

## Failure Cases

The CNN and PINN probe folders remain part of the paper package because they show where BCDO is still weak. The reference CNN branch contains useful conv-aware ideas, but the public accepted line is not a CNN-leading optimizer. The PINN probe likewise shows closure-friendly baselines ahead.

## Limitations

The current claim is deliberately narrow. BCDO is not a universal optimizer, not a dominant CNN optimizer, and not a PINN optimizer based on the checked-in evidence. The accepted public line is also slower than the strongest simple baselines.

## Reproducibility Statement

The repository includes focused tests, small runnable examples, a `build_paper_artifacts.py` script that regenerates the paper tables and figures from local CSVs, and a `run_paper_smoke.py` script that checks the paper package without rerunning the full expensive suite.

## Conclusion

The current repository supports a paper-scale draft only if the claim stays narrow. BCDO is best described as a blockwise direction-selection optimizer for workloads where the raw gradient direction is not trustworthy block-for-block. That is enough to make the method interesting. It is not enough to justify a broad default-optimizer story.

## Figures and Tables

- Accepted snapshot summary: `paper/tables/accepted_bcdo_summary.md`
- Accepted vs runnable split: `paper/tables/accepted_vs_runnable_summary.md`
- Reference CNN branch summary: `paper/tables/reference_cnn_summary.md`
- CNN probe summary: `paper/tables/cnn_probe_summary.md`
- PINN probe summary: `paper/tables/pinn_probe_summary.md`
- Ablation summary: `paper/tables/accepted_ablation_summary.md`
- Figures: `paper/figures/`

## References

- Bottou, Léon. “Large-Scale Machine Learning with Stochastic Gradient Descent.” 2010. <https://leon.bottou.org/papers/bottou-2010>
- Sutskever, Ilya, James Martens, George Dahl, and Geoffrey Hinton. “On the Importance of Initialization and Momentum in Deep Learning.” ICML 2013. <https://proceedings.mlr.press/v28/sutskever13.html>
- Hinton, Geoffrey. “Neural Networks for Machine Learning, Lecture 6e.” 2012. <https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf>
- Kingma, Diederik P., and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” 2015. <https://arxiv.org/abs/1412.6980>
- Loshchilov, Ilya, and Frank Hutter. “Decoupled Weight Decay Regularization.” 2019. <https://arxiv.org/abs/1711.05101>
- Gupta, Vineet, Tomer Koren, and Yoram Singer. “Shampoo: Preconditioned Stochastic Tensor Optimization.” 2018. <https://arxiv.org/abs/1802.09568>
- Martens, James, and Roger Grosse. “Optimizing Neural Networks with Kronecker-factored Approximate Curvature.” 2015. <https://arxiv.org/abs/1503.05671>
- Foret, Pierre, et al. “Sharpness-Aware Minimization for Efficiently Improving Generalization.” 2021. <https://arxiv.org/abs/2010.01412>
- Yu, Tianhe, et al. “Gradient Surgery for Multi-Task Learning.” 2020. <https://arxiv.org/abs/2001.06782>
"""
    output_path = PAPER_DIR / "bcdo_draft.md"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def build_paper_artifacts() -> dict[str, Path]:
    _ensure_dirs()

    accepted_benchmark_path = ACCEPTED_DIR / "benchmark_results.csv"
    accepted_ablation_path = ACCEPTED_DIR / "ablation_results.csv"
    current_benchmark_path = MAINLINE_DIR / "benchmark_results.csv"
    reference_benchmark_path = REFERENCE_CNN_DIR / "benchmark_results.csv"
    cnn_probe_path = CNN_PROBE_DIR / "benchmark_results.csv"
    pinn_probe_path = PINN_PROBE_DIR / "benchmark_results.csv"

    accepted_raw = _read_csv(accepted_benchmark_path)
    accepted_agg = aggregate_results(accepted_raw)
    current_raw = _read_csv(current_benchmark_path)
    current_agg = aggregate_results(current_raw)
    reference_raw = _read_csv(reference_benchmark_path)
    reference_agg = aggregate_results(reference_raw)
    cnn_probe_raw = _read_csv(cnn_probe_path)
    cnn_probe_agg = aggregate_results(cnn_probe_raw)
    pinn_probe_raw = _read_csv(pinn_probe_path)
    pinn_probe_agg = aggregate_results(pinn_probe_raw)
    accepted_ablation = _read_csv(accepted_ablation_path)

    accepted_table = _accepted_table(accepted_agg)
    current_vs_accepted = _current_vs_accepted_table(accepted_agg, current_agg)
    runtime_table = _runtime_table(accepted_agg)
    reference_table = _reference_cnn_table(reference_agg)
    cnn_probe_table = _reference_cnn_table(cnn_probe_agg)
    pinn_table = _pinn_table(pinn_probe_agg)
    accepted_ablation_table = _ablation_table(accepted_ablation)

    accepted_win_summary = _summarize_wins(
        accepted_agg,
        "blockwise_consensus_direction_optimizer",
        ["bcdo_mainline_legacy", "bcdo_cnn_reference", "adamw", "rmsprop", "sgd_momentum", "magneto_hamiltonian_adam"],
        "accepted_public_snapshot",
        accepted_benchmark_path,
    )

    _write_table(
        "accepted_bcdo_summary",
        accepted_table,
        [accepted_benchmark_path],
        "Accepted first-release BCDO rows aggregated from the checked-in public snapshot.",
    )
    _write_table(
        "accepted_vs_runnable_summary",
        current_vs_accepted,
        [accepted_benchmark_path, current_benchmark_path],
        "Separation between the fixed accepted public snapshot and the current writable runnable output path.",
    )
    _write_table(
        "accepted_win_summary",
        accepted_win_summary,
        [accepted_benchmark_path],
        "Meaningful win counts for the accepted public snapshot, computed from the checked-in benchmark CSV.",
    )
    _write_table(
        "accepted_ablation_summary",
        accepted_ablation_table,
        [accepted_ablation_path],
        "Accepted public ablation summary for the mainline BCDO design.",
    )
    _write_table(
        "reference_cnn_summary",
        reference_table,
        [reference_benchmark_path],
        "Reference CNN branch summary kept visible as benchmark context rather than as the public identity of the repo.",
    )
    _write_table(
        "cnn_probe_summary",
        cnn_probe_table,
        [cnn_probe_path],
        "Focused CNN probe used to keep the CNN gap visible.",
    )
    _write_table(
        "pinn_probe_summary",
        pinn_table,
        [pinn_probe_path],
        "Focused PINN probe used to keep closure-friendly failure cases visible.",
    )
    _write_table(
        "runtime_summary",
        runtime_table,
        [accepted_benchmark_path],
        "Runtime summary aggregated from the accepted public snapshot.",
    )

    _plot_grouped_bar(
        accepted_table[accepted_table["optimizer"].isin(["blockwise_consensus_direction_optimizer", "adamw", "rmsprop", "sgd_momentum"])],
        "task",
        "mean_best_val_loss",
        "optimizer",
        "Accepted BCDO Best Validation Loss",
        FIGURES_DIR / "accepted_best_val_loss_by_task.png",
        "mean best val loss",
    )
    _plot_simple_bar(
        runtime_table[runtime_table["optimizer"].isin(["blockwise_consensus_direction_optimizer", "bcdo_mainline_legacy", "bcdo_cnn_reference", "adamw", "rmsprop", "sgd_momentum"])],
        "optimizer",
        "mean_runtime_per_step_ms",
        "Accepted BCDO Runtime Per Step",
        FIGURES_DIR / "runtime_per_step_comparison.png",
        "ms per step",
    )
    _plot_simple_bar(
        accepted_win_summary,
        "baseline",
        "meaningful_wins",
        "Accepted BCDO Win Counts",
        FIGURES_DIR / "accepted_win_summary.png",
        "meaningful wins",
    )
    _plot_simple_bar(
        accepted_ablation_table.head(10),
        "variant_name",
        "mean_selection_score" if "mean_selection_score" in accepted_ablation_table.columns else "mean_best_val_loss",
        "Accepted BCDO Ablation Impact",
        FIGURES_DIR / "ablation_impact_plot.png",
        "mean selection score" if "mean_selection_score" in accepted_ablation_table.columns else "mean best val loss",
    )
    _plot_grouped_bar(
        cnn_probe_table[cnn_probe_table["task"].isin(["digits_cnn", "digits_mlp"])],
        "task",
        "mean_best_val_accuracy",
        "optimizer",
        "BCDO CNN Probe",
        FIGURES_DIR / "cnn_gap_plot.png",
        "mean best val accuracy",
    )
    _plot_grouped_bar(
        pinn_table,
        "task",
        "mean_best_val_loss",
        "optimizer",
        "BCDO PINN Probe",
        FIGURES_DIR / "pinn_probe_failure_plot.png",
        "mean best val loss",
    )

    summary_rows: list[dict[str, Any]] = []
    for baseline in ["bcdo_mainline_legacy", "bcdo_cnn_reference", "adamw", "rmsprop", "sgd_momentum", "magneto_hamiltonian_adam"]:
        subset = accepted_win_summary[accepted_win_summary["baseline"] == baseline]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "section": "accepted_public_snapshot",
                "metric": f"meaningful_wins_vs_{baseline}",
                "value": int(subset["meaningful_wins"].iloc[0]),
                "source_csv": _relative(accepted_benchmark_path),
                "note": "computed from aggregate_results and compute_meaningful_wins",
            }
        )
    accepted_best_row = accepted_agg[accepted_agg["optimizer"] == "blockwise_consensus_direction_optimizer"].sort_values(
        ["mean_best_val_accuracy", "mean_best_val_loss"], ascending=[False, True]
    ).iloc[0]
    summary_rows.append(
        {
            "section": "accepted_public_snapshot",
            "metric": "best_bcdo_task",
            "value": accepted_best_row["task"],
            "source_csv": _relative(accepted_benchmark_path),
            "note": f"best val loss {_fmt(accepted_best_row['mean_best_val_loss'], 6)}, best val accuracy {_fmt(accepted_best_row['mean_best_val_accuracy'], 6)}",
        }
    )
    for optimizer_name in ["blockwise_consensus_direction_optimizer", "bcdo_mainline_legacy", "bcdo_cnn_reference", "adamw", "rmsprop", "sgd_momentum"]:
        subset = runtime_table[runtime_table["optimizer"] == optimizer_name]
        if subset.empty:
            continue
        summary_rows.append(
            {
                "section": "runtime",
                "metric": f"{optimizer_name}_mean_runtime_per_step_ms",
                "value": float(subset["mean_runtime_per_step_ms"].iloc[0]),
                "source_csv": _relative(accepted_benchmark_path),
                "note": "mean over accepted public snapshot tasks",
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_path = PAPER_DIR / "bcdo_results_summary.csv"
    summary_frame.to_csv(summary_path, index=False)

    claims_path = _write_claims_audit(accepted_win_summary, cnn_probe_agg, pinn_probe_agg, runtime_table)
    draft_path = _write_paper_draft(accepted_best_row, accepted_win_summary, runtime_table)

    manifest = {
        "summary_csv": _relative(summary_path),
        "claims_audit": _relative(claims_path),
        "draft": _relative(draft_path),
        "tables_dir": _relative(TABLES_DIR),
        "figures_dir": _relative(FIGURES_DIR),
    }
    (PAPER_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "summary_csv": summary_path,
        "claims_audit": claims_path,
        "draft": draft_path,
        "tables_dir": TABLES_DIR,
        "figures_dir": FIGURES_DIR,
    }
