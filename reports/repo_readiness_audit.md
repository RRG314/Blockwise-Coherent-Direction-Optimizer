# Repo Readiness Audit

## Scope

This audit covers the standalone Blockwise Consensus Direction Optimizer repository in:

- `/Users/stevenreid/Documents/New project/repos/block-direction-optimizer`

The goal is to verify that a fresh clone can install the repo, import the optimizer, run a smoke test, inspect accepted results, and reproduce the documented report flow without depending on unrelated workspace repositories.

This audit also checks whether the public documentation matches the real repository layout and whether the public docs now point to one checked literature index.

## Inventory summary

### Optimizer implementation files

- `src/optimizers/block_direction_optimizer_v4_fast.py`
- `src/optimizers/blockwise_consensus_direction_optimizer.py`
- `src/optimizers/block_direction_optimizer_v42.py`
- earlier lineage/support modules in `src/optimizers/`

### Public import paths

- `from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer`
- `from optimizers import BlockwiseConsensusDirectionOptimizer, BCDO`
- backward-compatible import:
  - `from optimizers import BlockDirectionOptimizerV4Fast`

### Benchmark and report scripts

- `scripts/run_block_direction_v4_fast_smoke.py`
- `scripts/run_block_direction_v4_fast_tuning.py --config configs/block_direction_v4_fast_tuning.yaml`
- `scripts/run_block_direction_v4_fast_benchmarks.py --config configs/block_direction_v4_fast_default.yaml`
- `scripts/run_block_direction_v4_fast_ablation.py --config configs/block_direction_v4_fast_ablation.yaml`
- `scripts/export_block_direction_v4_fast_report.py`
- public wrappers:
  - `scripts/run_bcdo_smoke.py`
  - `scripts/run_bcdo_benchmarks.py --config configs/block_direction_v4_fast_default.yaml`
  - `scripts/export_bcdo_report.py`

### Config files

- mainline:
  - `configs/block_direction_v4_fast_default.yaml`
  - `configs/block_direction_v4_fast_tuning.yaml`
  - `configs/block_direction_v4_fast_ablation.yaml`
- probes:
  - `configs/block_direction_v4_fast_cnn_probe.yaml`
  - `configs/block_direction_v4_fast_cnn_tuning.yaml`
  - `configs/block_direction_v4_fast_gpu.yaml`
  - `configs/block_direction_v4_fast_pinn_probe.yaml`
  - `configs/block_direction_v4_fast_pinn_tuning.yaml`

### Tests

- `tests/test_block_direction_v4_fast.py`
- `tests/test_blockwise_consensus_direction_optimizer.py`
- `tests/test_block_direction_benchmark_outputs.py`
- reference branch test retained:
  - `tests/test_block_direction_v42.py`

### Reports and result CSVs

- accepted public snapshot:
  - `reports/accepted_bcdo/`
- runnable mainline output path:
  - `reports/bcdo_mainline/`
- reference CNN branch:
  - `reports/reference_cnn_branch/`
- probes:
  - `reports/bcdo_cnn_probe/`
  - `reports/bcdo_pinn_probe/`
  - `reports/bcdo_mps_probe/`

### Dependency file

- `pyproject.toml`
- literature and external optimizer references:
  - `REFERENCES.md`
- comparison and method notes:
  - `docs/METHOD.md`
  - `docs/COMPARISONS.md`

## Readiness answers

### Can someone install this repo?

Yes.

Use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Can someone import the optimizer?

Yes.

Public import:

```python
from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer
```

Backward-compatible import:

```python
from optimizers import BlockDirectionOptimizerV4Fast
```

### Can someone run a smoke test?

Yes.

```bash
python scripts/run_block_direction_v4_fast_smoke.py
```

or

```bash
python scripts/run_bcdo_smoke.py
```

### Can someone reproduce the accepted benchmark summary?

Yes, with an important distinction:

- `reports/accepted_bcdo/` is the accepted first-release snapshot.
- `reports/bcdo_mainline/` is the runnable output path used by current scripts.

The README cites the accepted snapshot, while the runnable scripts target the runnable mainline folder.

### Can someone inspect results?

Yes.

Accepted results are visible directly in:

- `reports/accepted_bcdo/benchmark_results.csv`
- `reports/accepted_bcdo/final_report.md`
- `reports/accepted_bcdo/best_by_task.csv`
- `reports/accepted_bcdo/win_flags.csv`

### Are CPU, GPU, and MPS paths supported?

Partially.

- CPU: supported and used for the accepted public benchmark line.
- MPS: supported by focused tests and the probe config.
- CUDA: supported conditionally in focused tests when CUDA is available.

The repo does not claim accepted benchmark numbers from GPU runs.

### Are stale failed experiments clearly separated?

Yes.

Historical summary files that do not define the current accepted public line were moved to:

- `reports/block_direction_experimental_backup_20260429/`

### Are there stale or misleading claims?

Public-facing docs were rewritten to avoid:

- universal-superiority claims
- state-of-the-art language
- public-first naming around internal V4 lineage

The public docs now also point to one checked reference index:

- `REFERENCES.md`

The repo still keeps internal class and config names for compatibility, but the README now treats them as implementation details rather than branding.

## Missing pieces and limits

- CNN performance is still weaker than strong baselines such as RMSProp on the accepted suite.
- Runtime overhead is still significantly higher than SGD, RMSProp, and AdamW.
- PINN probes are present, but this is not currently a winning PINN optimizer story.
- The reference CNN branch remains separate because its strongest ideas are not yet cleanly absorbed into the accepted public mainline.

## Verdict

This repository is now clone-and-run ready for its current research scope:

- installable
- importable through a clean public name
- smoke-testable
- benchmark-report reproducible
- equipped with accepted public artifacts and separated exploratory history
