# BCDO Complete System, Inventory, and Results Report

Prepared: `2026-04-30`

This document is the all-in-one reference for the current Blockwise Consensus Direction Optimizer repository. It combines the system view, the file and artifact inventory, and the accepted benchmark interpretation into one place so the repo can be understood without hopping across multiple audit and report files.

The public optimizer name in this repository is `BlockwiseConsensusDirectionOptimizer`, with `BCDO` used as the short name in code, configs, and reports. The accepted first-release mainline is the blockwise direction-selection branch documented in [../README.md](../README.md) and implemented in [../src/optimizers/blockwise_consensus_direction_optimizer.py](../src/optimizers/blockwise_consensus_direction_optimizer.py).

## 1. System Overview

The repository is organized around one public optimizer surface and one reference branch. The public surface is the accepted BCDO mainline. The reference branch, `BCDOCNNReference`, remains in the repo because it is part of the empirical story around conv-aware refinements, but it is not the accepted first-release identity.

The system has five practical layers.

The package layer contains the importable optimizer code under [../src/optimizers](../src/optimizers). The research harness under [../src/optimizer_research](../src/optimizer_research) provides tasks, baselines, reporting, and suite definitions. The scripts layer under [../scripts](../scripts) turns those suites into runnable smoke, tuning, benchmark, ablation, and export commands. The config layer under [../configs](../configs) fixes the benchmark and tuning surfaces. The report layer in this `reports/` directory stores accepted artifacts, runnable outputs, and focused probes.

That separation matters because the accepted public benchmark line is not identical to the current writable output directory. The accepted first-release snapshot lives in [accepted_bcdo](accepted_bcdo), while the current runnable output path used by the scripts is [bcdo_mainline](bcdo_mainline).

## 2. Public Identity and Code Surface

The public import path is:

```python
from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer
```

The package export layer in [../src/optimizers/__init__.py](../src/optimizers/__init__.py) also exposes the short alias:

```python
from optimizers import BCDO, BlockwiseConsensusDirectionOptimizer
```

The accepted mainline code path is:

- [../src/optimizers/blockwise_consensus_direction_optimizer.py](../src/optimizers/blockwise_consensus_direction_optimizer.py)

The reference CNN branch is:

- [../src/optimizers/bcdo_cnn_reference.py](../src/optimizers/bcdo_cnn_reference.py)

The shared structured logic that supports the public optimizer is in:

- [../src/optimizers/bcdo_direction_selection_base.py](../src/optimizers/bcdo_direction_selection_base.py)
- [../src/optimizers/bcdo_structured_core.py](../src/optimizers/bcdo_structured_core.py)
- [../src/optimizers/diagnostics.py](../src/optimizers/diagnostics.py)
- [../src/optimizers/optimizer_utils.py](../src/optimizers/optimizer_utils.py)

The repo still contains additional optimizer modules inherited from the wider research workspace. They remain because the comparison harness and some historical reports depend on them. They are not the public identity of this repo.

## 3. Research Harness Inventory

The main research-harness files are:

- [../src/optimizer_research/bcdo_suite.py](../src/optimizer_research/bcdo_suite.py)
- [../src/optimizer_research/bcdo_cnn_reference_suite.py](../src/optimizer_research/bcdo_cnn_reference_suite.py)
- [../src/optimizer_research/tasks.py](../src/optimizer_research/tasks.py)
- [../src/optimizer_research/baselines.py](../src/optimizer_research/baselines.py)
- [../src/optimizer_research/benchmarking.py](../src/optimizer_research/benchmarking.py)
- [../src/optimizer_research/reporting.py](../src/optimizer_research/reporting.py)
- [../src/optimizer_research/config.py](../src/optimizer_research/config.py)
- [../src/optimizer_research/bcdo_literature.py](../src/optimizer_research/bcdo_literature.py)

In practical terms, these files do three jobs. They define the benchmark tasks and baseline optimizer registry, they run structured benchmark loops with consistent output schemas, and they export the markdown and CSV artifacts checked into this repo.

## 4. Script Inventory

The public mainline scripts are:

- [../scripts/run_bcdo_smoke.py](../scripts/run_bcdo_smoke.py)
- [../scripts/run_bcdo_tuning.py](../scripts/run_bcdo_tuning.py)
- [../scripts/run_bcdo_benchmarks.py](../scripts/run_bcdo_benchmarks.py)
- [../scripts/run_bcdo_ablation.py](../scripts/run_bcdo_ablation.py)
- [../scripts/export_bcdo_report.py](../scripts/export_bcdo_report.py)

The reference CNN branch scripts are:

- [../scripts/run_bcdo_cnn_reference_smoke.py](../scripts/run_bcdo_cnn_reference_smoke.py)
- [../scripts/run_bcdo_cnn_reference_tuning.py](../scripts/run_bcdo_cnn_reference_tuning.py)
- [../scripts/run_bcdo_cnn_reference_benchmarks.py](../scripts/run_bcdo_cnn_reference_benchmarks.py)
- [../scripts/run_bcdo_cnn_reference_ablation.py](../scripts/run_bcdo_cnn_reference_ablation.py)
- [../scripts/export_bcdo_cnn_reference_report.py](../scripts/export_bcdo_cnn_reference_report.py)

The paper-facing helper scripts are:

- [../scripts/build_paper_artifacts.py](../scripts/build_paper_artifacts.py)
- [../scripts/run_paper_smoke.py](../scripts/run_paper_smoke.py)

Together, these scripts cover the accepted mainline, the current runnable mainline, and the conv-aware reference branch. There is no notebook in the BCDO repo at the moment; the reproducibility surface is script-first rather than notebook-first.

## 5. Config Inventory

The accepted and runnable mainline configs are:

- [../configs/bcdo_default.yaml](../configs/bcdo_default.yaml)
- [../configs/bcdo_tuning.yaml](../configs/bcdo_tuning.yaml)
- [../configs/bcdo_ablation.yaml](../configs/bcdo_ablation.yaml)

The reference and probe configs are:

- [../configs/bcdo_cnn_reference_default.yaml](../configs/bcdo_cnn_reference_default.yaml)
- [../configs/bcdo_cnn_reference_tuning.yaml](../configs/bcdo_cnn_reference_tuning.yaml)
- [../configs/bcdo_cnn_reference_ablation.yaml](../configs/bcdo_cnn_reference_ablation.yaml)
- [../configs/bcdo_cnn_probe.yaml](../configs/bcdo_cnn_probe.yaml)
- [../configs/bcdo_cnn_tuning.yaml](../configs/bcdo_cnn_tuning.yaml)
- [../configs/bcdo_pinn_probe.yaml](../configs/bcdo_pinn_probe.yaml)
- [../configs/bcdo_pinn_tuning.yaml](../configs/bcdo_pinn_tuning.yaml)
- [../configs/bcdo_gpu.yaml](../configs/bcdo_gpu.yaml)

These configs are part of the documented empirical story. The accepted public line is CPU-oriented, while MPS and GPU-related paths are treated as probes rather than as the basis for the public benchmark claim.

## 6. Test Inventory

The focused repo-readiness tests are:

- [../tests/test_bcdo.py](../tests/test_bcdo.py)
- [../tests/test_blockwise_consensus_direction_optimizer.py](../tests/test_blockwise_consensus_direction_optimizer.py)
- [../tests/test_bcdo_benchmark_outputs.py](../tests/test_bcdo_benchmark_outputs.py)
- [../tests/test_bcdo_cnn_reference.py](../tests/test_bcdo_cnn_reference.py)

These tests cover imports, initialization, one-step parameter motion, no-NaN smoke behavior, diagnostics access, candidate-selection execution, winner-take-all behavior, convolutional tensor handling, matrix-consensus handling, and benchmark-output schemas.

## 7. Documentation Inventory

The top-level package and documentation files that define the repo surface are:

- [../README.md](../README.md)
- [../REPRODUCING.md](../REPRODUCING.md)
- [../REFERENCES.md](../REFERENCES.md)
- [../ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md)
- [../LICENSE](../LICENSE)
- [../pyproject.toml](../pyproject.toml)
- [../CITATION.cff](../CITATION.cff)
- [../docs/CLAIM.md](../docs/CLAIM.md)
- [../docs/METHOD.md](../docs/METHOD.md)
- [../docs/COMPARISONS.md](../docs/COMPARISONS.md)
- [../docs/CNN_REFERENCE.md](../docs/CNN_REFERENCE.md)
- [../docs/FAILURE_CASES.md](../docs/FAILURE_CASES.md)
- [repo_readiness_audit.md](repo_readiness_audit.md)
- [repo_update_report.md](repo_update_report.md)
- [../paper/bcdo_draft.md](../paper/bcdo_draft.md)
- [../paper/bcdo_claims_audit.md](../paper/bcdo_claims_audit.md)

The bibliography and external optimizer references are centralized in [../REFERENCES.md](../REFERENCES.md). That file is the canonical reference index for BCDO’s comparisons with SGD, momentum, RMSProp, AdamW, Muon, Shampoo, K-FAC, SAM, and the other named external methods that appear in the repo.

The current runnable example layer now also includes:

- [../examples/basic_usage.py](../examples/basic_usage.py)
- [../examples/bcdo_minimal_mlp.py](../examples/bcdo_minimal_mlp.py)
- [../examples/bcdo_blockwise_direction_demo.py](../examples/bcdo_blockwise_direction_demo.py)
- [../examples/bcdo_compare_against_adamw.py](../examples/bcdo_compare_against_adamw.py)

## 8. Report and Artifact Inventory

The checked-in report families are:

- [accepted_bcdo](accepted_bcdo)
- [bcdo_mainline](bcdo_mainline)
- [reference_cnn_branch](reference_cnn_branch)
- [bcdo_cnn_probe](bcdo_cnn_probe)
- [bcdo_pinn_probe](bcdo_pinn_probe)
- [bcdo_mps_probe](bcdo_mps_probe)
- [../paper](../paper)

Each of the main report families contains some or all of the following artifact types:

- `benchmark_results.csv` for per-run or per-seed benchmark rows
- `best_by_task.csv` for task-level winners
- `win_flags.csv` for pairwise comparison flags
- `smoke_results.csv` for smoke-run output where applicable
- `tuning_results.csv` for parameter-search results where applicable
- `ablation_results.csv` for feature-removal comparisons where applicable
- `current_state.md` for current run interpretation
- `final_report.md` for the human-facing report summary where applicable
- `math_definition.md` for the exported method summary
- `literature_scan.md` and `literature_matrix.csv` for the report-local literature mapping
- `figures/` for plots exported by the reporting layer

The main distinction to keep straight is this: [accepted_bcdo](accepted_bcdo) is the accepted public benchmark line for first release, while [bcdo_mainline](bcdo_mainline) is the writable output target used by the current smoke, tuning, benchmark, ablation, and export scripts.

## 9. Accepted Public Results

The accepted public result source is [accepted_bcdo](accepted_bcdo), especially:

- [accepted_bcdo/final_report.md](accepted_bcdo/final_report.md)
- [accepted_bcdo/benchmark_results.csv](accepted_bcdo/benchmark_results.csv)
- [accepted_bcdo/best_by_task.csv](accepted_bcdo/best_by_task.csv)
- [accepted_bcdo/win_flags.csv](accepted_bcdo/win_flags.csv)

For public-facing summary numbers, the repository currently treats [accepted_bcdo/final_report.md](accepted_bcdo/final_report.md) and the top-level README as the authoritative first-release summary. The CSV files remain the authoritative row-level artifact source. That distinction matters because the CSVs store per-seed rows, while the accepted report surfaces mean or selected summary values.

The accepted first-release summary is:

- strongest standard row: `breast_cancer_mlp`, mean best val loss `0.061726`, mean best val accuracy `0.985677`
- strongest additional dense row: `wine_mlp`, mean best val loss `0.058005`, mean best val accuracy `0.985185`
- strongest MLP multi-class row: `digits_mlp`, mean best val loss `0.079455`, mean best val accuracy `0.979818`
- strongest accepted CNN row in the mainline snapshot: `digits_cnn`, mean best val loss `0.814210`, mean best val accuracy `0.740294`
- strongest stress rows:
  - `oscillatory_valley`: `-0.071087`
  - `plateau_escape_objective`: `-1.050024`
  - `saddle_objective`: `-4.132231`

The accepted competitive summary is:

- wins vs `bcdo_mainline_legacy`: `4`
- wins vs `BCDOCNNReference`: `4`
- wins vs `AdamW`: `7`
- tracked `2x` wins vs `AdamW`: `5`
- wins vs `RMSProp`: `6`
- wins vs `SGD+momentum`: `4`
- wins vs the internal Coherent Momentum baseline family: `10`

The accepted runtime summary is:

- mean BCDO runtime per step: `24.0797 ms`
- mean legacy fast-path runtime per step: `23.8415 ms`
- mean reference CNN-branch runtime per step: `22.8265 ms`

The accepted ablation story is also clear. Typed conv/dense split, cheap conv structure support, matrix consensus, block structure, stable consensus, and trusted-direction memory all help the accepted line. Recoverability periodic gating hurts the accepted line and is not part of the public default path.

## 10. Runnable Mainline and Probe Results

The current runnable mainline lives in [bcdo_mainline](bcdo_mainline). It exists so a fresh clone can produce current smoke, tuning, benchmark, and export outputs without overwriting the accepted public snapshot. The accepted public line and the current runnable output path therefore serve different purposes.

The reference CNN branch in [reference_cnn_branch](reference_cnn_branch) is the most important secondary result family because it documents what happened when conv-aware trust and conv-safe logic were explored more aggressively. Its final report shows that the branch improved some standard and CNN-side behavior, but did not displace strong baselines cleanly enough to become the accepted first-release public identity.

The probe folders serve narrower roles. [bcdo_cnn_probe](bcdo_cnn_probe) exists to test CNN behavior beyond the accepted mainline. [bcdo_pinn_probe](bcdo_pinn_probe) exists to show that BCDO is not currently a winning PINN story. [bcdo_mps_probe](bcdo_mps_probe) exists to verify compatibility and collect exploratory MPS behavior without turning MPS into a public performance-claim platform.

## 11. Practical Interpretation

The repository supports a narrow but credible interpretation. BCDO is a blockwise direction-selection optimizer whose strongest case is not “better adaptive scaling,” but “better blockwise direction choice when raw gradient direction is unstable or structurally misleading.”

That is why the accepted results matter more on the stress-oriented and structure-sensitive slices than on a broad winner-take-all task count. The repo does not support a universal optimizer claim. It does not support a broad claim that BCDO replaces RMSProp or SGD with momentum. It also does not currently support a broad CNN leadership claim or a PINN leadership claim.

The best public reading is that BCDO is a structured specialist/generalist hybrid. It has a real mechanism, a reproducible accepted benchmark story, and clear evidence about what helped and what did not. It is also still slower than the strongest simple baselines and still incomplete on CNN-heavy and PINN-heavy fronts.

## 12. Reproduction Entry Points

The shortest reproducibility path is:

- install from [../pyproject.toml](../pyproject.toml)
- run the focused tests listed in [../README.md](../README.md)
- run [../scripts/run_bcdo_smoke.py](../scripts/run_bcdo_smoke.py)
- run [../scripts/run_bcdo_benchmarks.py](../scripts/run_bcdo_benchmarks.py) with [../configs/bcdo_default.yaml](../configs/bcdo_default.yaml)
- run [../scripts/export_bcdo_report.py](../scripts/export_bcdo_report.py)

For the accepted public benchmark line, read [accepted_bcdo/final_report.md](accepted_bcdo/final_report.md). For exact reproducibility instructions, use [../REPRODUCING.md](../REPRODUCING.md).

## 13. Current Repository Status

The repo is clone-and-run ready for its present scope. It has a stable public import path, focused tests, runnable scripts, checked references, accepted public artifacts, a separated reference CNN branch, and a clear division between accepted public results and current writable output directories.

The main current gaps are practical rather than organizational. Runtime is still higher than the strongest simple baselines. CNN behavior remains only partially absorbed into the accepted mainline. PINN probes remain negative or inconclusive. Those limits are visible in the checked-in reports and are part of the honest public story of the repo.
