# Repo Readiness Audit

## Scope

This audit covers the standalone Blockwise Consensus Direction Optimizer repository in:

- `/Users/stevenreid/Documents/New project/repos/block-direction-optimizer`

The goal is to verify that a fresh clone can install the repo, import the optimizer, run a smoke test, inspect accepted results, and reproduce the documented report flow without depending on unrelated workspace repositories.

This audit also checks whether the public documentation matches the real repository layout and whether the public docs now point to one checked literature index.

## Inventory summary

### Optimizer implementation files

- `src/optimizers/blockwise_consensus_direction_optimizer.py`
- `src/optimizers/bcdo_cnn_reference.py`
- earlier lineage/support modules in `src/optimizers/`

### Public import paths

- `from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer`
- `from optimizers import BlockwiseConsensusDirectionOptimizer, BCDO`

### Benchmark and report scripts

- `scripts/run_bcdo_smoke.py`
- `scripts/run_bcdo_tuning.py --config configs/bcdo_tuning.yaml`
- `scripts/run_bcdo_benchmarks.py --config configs/bcdo_default.yaml`
- `scripts/run_bcdo_ablation.py --config configs/bcdo_ablation.yaml`
- `scripts/export_bcdo_report.py`
- public wrappers:
  - `scripts/run_bcdo_smoke.py`
  - `scripts/run_bcdo_benchmarks.py --config configs/bcdo_default.yaml`
  - `scripts/export_bcdo_report.py`

### Config files

- mainline:
  - `configs/bcdo_default.yaml`
  - `configs/bcdo_tuning.yaml`
  - `configs/bcdo_ablation.yaml`
- probes:
  - `configs/bcdo_cnn_probe.yaml`
  - `configs/bcdo_cnn_tuning.yaml`
  - `configs/bcdo_gpu.yaml`
  - `configs/bcdo_pinn_probe.yaml`
  - `configs/bcdo_pinn_tuning.yaml`

### Tests

- `tests/test_bcdo.py`
- `tests/test_blockwise_consensus_direction_optimizer.py`
- `tests/test_bcdo_benchmark_outputs.py`
- reference branch test retained:
  - `tests/test_bcdo_cnn_reference.py`

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
from optimizers import BlockwiseConsensusDirectionOptimizer
```

### Can someone run a smoke test?

Yes.

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

The public report surface is split into accepted, runnable-mainline, and probe folders. Older exploratory material is not part of the accepted first-release line.

### Are there stale or misleading claims?

The main public docs were rewritten to avoid universal-superiority claims, state-of-the-art language, and public-first naming around internal lineage labels. They also now point consistently to one checked reference index:

- `REFERENCES.md`

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
