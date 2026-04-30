# Blockwise Consensus Direction Optimizer

A PyTorch optimizer that selects update directions at the block level instead of transforming a single gradient direction.

## Overview

Most optimizers start from one gradient direction and then smooth it, rescale it, or precondition it. BCDO takes a different route: for each parameter block it builds a small set of candidate directions, scores them with structural trust signals, chooses the most trusted direction with winner-take-all selection, and applies a bounded blockwise step.

This is a specialist optimizer for regimes where raw gradient direction is unstable, conflicting, or structurally misleading. It is not a universal replacement for standard optimizers.

The public optimizer name in this repository is `BlockwiseConsensusDirectionOptimizer`, with `BCDO` as the short alias used in code and reports.

The repository also keeps one reference CNN branch:

- `BCDOCNNReference` (documented in [docs/CNN_REFERENCE.md](docs/CNN_REFERENCE.md))

The public explanation of the method lives in [docs/METHOD.md](docs/METHOD.md), the side-by-side comparison notes live in [docs/COMPARISONS.md](docs/COMPARISONS.md), and the external literature used throughout the repo is collected in [REFERENCES.md](REFERENCES.md). The repository-level attribution note is in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

## Relation to existing optimizers

BCDO is easiest to understand by contrast with the optimizer families it is measured against. `SGD` follows the raw gradient directly and `SGD+momentum` replaces that raw direction with one smoothed velocity recursion; see [SGD and momentum](REFERENCES.md#sgd-and-momentum). `RMSProp` keeps the basic gradient direction but rescales it coordinate-wise with squared-gradient history; see [RMSProp](REFERENCES.md#rmsprop). `Adam` and `AdamW` combine momentum and adaptive scaling in a single per-parameter update; see [Adam and AdamW](REFERENCES.md#adam-and-adamw). `Muon`, `Shampoo`, and `K-FAC` use matrix or tensor structure more aggressively, but they do so through orthogonalization or preconditioning rather than through explicit candidate-direction choice; see [Muon](REFERENCES.md#muon), [Shampoo](REFERENCES.md#shampoo), and [K-FAC](REFERENCES.md#k-fac).

BCDO is different in a narrower and more concrete way. It does not start from one gradient direction and then keep modifying that direction. It forms a small candidate set per block, scores the candidates, selects one winner, and only then decides how far to move. That is why this repository treats BCDO as a direction-selection optimizer rather than as another Adam variant with extra controls.

## Method

The accepted public mainline is implemented in [src/optimizers/blockwise_consensus_direction_optimizer.py](src/optimizers/blockwise_consensus_direction_optimizer.py). The public alias is thin by design; the algorithm lives in one implementation so that the code path used in the accepted reports is the same code path a newcomer imports.

### Block formation

BCDO uses an internal smart block-formation rule:

- vectors and small matrices stay as tensor blocks
- larger matrices use row-level blocks
- convolutional tensors are handled through a typed profile split

The implementation key for this grouping rule is still called `smart_bcdo` in the code and configs. That name is retained for compatibility, but it is not part of the public method name.

### Candidate set

Per block, BCDO scores a small candidate set:

- `gradient`
- `stable_consensus`
- `trusted_direction`
- `low_rank_matrix`

### Trust score

Each candidate is scored using:

- descent alignment
- memory coherence
- quality memory
- norm stability
- consensus support
- oscillation penalty
- conflict penalty
- cost penalty

### Selection rule

BCDO uses winner-take-all selection:

`d_i* = argmax_d T_i(d)`

### Step rule

The selected block direction is scaled with a bounded block-energy rule:

`alpha_i = lr * trust_i * ||g_i|| / (sqrt(E_i) + eps)^p`

Convolutional tensors use typed conv-safe scaling on top of the same direction-selection core. The accepted mainline deliberately keeps that convolutional logic on the scaling side rather than turning it into another full candidate generator, because earlier branch work did not justify the heavier alternative on the accepted benchmark line.

## Current implementation

Main files in this repository:

- [src/optimizers/blockwise_consensus_direction_optimizer.py](src/optimizers/blockwise_consensus_direction_optimizer.py)
- [src/optimizers/bcdo_cnn_reference.py](src/optimizers/bcdo_cnn_reference.py)
- [src/optimizer_research](src/optimizer_research)
- [tests](tests)
- [scripts](scripts)
- [configs](configs)
- [reports](reports)

If you want the shortest path through the repository, start with:

- [docs/METHOD.md](docs/METHOD.md) for the accepted algorithmic design,
- [docs/COMPARISONS.md](docs/COMPARISONS.md) for how it differs from the standard baselines,
- [REFERENCES.md](REFERENCES.md) for the literature used in those comparisons,
- [reports/accepted_bcdo/final_report.md](reports/accepted_bcdo/final_report.md) for the accepted benchmark summary.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Quick start

```python
import torch
from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer

model = torch.nn.Sequential(
    torch.nn.Linear(32, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
)

optimizer = BlockwiseConsensusDirectionOptimizer(model.parameters(), lr=1.5e-2)
criterion = torch.nn.CrossEntropyLoss()

x = torch.randn(16, 32)
y = torch.randint(0, 10, (16,))

optimizer.zero_grad(set_to_none=True)
loss = criterion(model(x), y)
optimizer.set_current_loss(loss.item())
loss.backward()
optimizer.step()

print(optimizer.latest_diagnostics())
```

Backward-compatible imports still work:

```python
from optimizers import BCDO, BlockwiseConsensusDirectionOptimizer
```

## Running tests

Focused repo-local tests:

```bash
pytest tests/test_bcdo.py -q
pytest tests/test_blockwise_consensus_direction_optimizer.py tests/test_bcdo_benchmark_outputs.py -q
```

These are the tests used as repo-readiness evidence. Full workspace tests are not used here because unrelated repositories in the parent workspace can fail independently.

## Running benchmarks

Mainline scripts:

```bash
python scripts/run_bcdo_smoke.py
python scripts/run_bcdo_tuning.py --config configs/bcdo_tuning.yaml
python scripts/run_bcdo_benchmarks.py --config configs/bcdo_default.yaml
python scripts/run_bcdo_ablation.py --config configs/bcdo_ablation.yaml
python scripts/export_bcdo_report.py
```

## Results

The accepted public benchmark snapshot is stored in [reports/accepted_bcdo](reports/accepted_bcdo).

Accepted summary from [reports/accepted_bcdo/benchmark_results.csv](reports/accepted_bcdo/benchmark_results.csv):

- wins vs [AdamW](REFERENCES.md#adam-and-adamw): `7`
- wins vs [RMSProp](REFERENCES.md#rmsprop): `6`
- wins vs [SGD+momentum](REFERENCES.md#sgd-and-momentum): `4`
- wins vs [the internal Coherent Momentum baseline family](REFERENCES.md#internal-comparison-baselines): `10`
- tracked `2x` wins vs [AdamW](REFERENCES.md#adam-and-adamw): `5`
- strongest standard row: `breast_cancer_mlp`, best val loss `0.061726`, best val acc `0.985677`
- strongest additional dense row: `wine_mlp`, best val loss `0.058005`, best val acc `0.985185`
- strongest MLP multi-class row: `digits_mlp`, best val loss `0.079455`, best val acc `0.979818`
- strongest accepted CNN row in the mainline snapshot: `digits_cnn`, best val loss `0.814210`, best val acc `0.740294`
- strongest stress rows:
  - `oscillatory_valley`: `-0.071087`
  - `plateau_escape_objective`: `-1.050024`
  - `saddle_objective`: `-4.132231`

Those results support a narrow claim, not a broad one. BCDO has a real edge on the accepted stress-oriented slice, and it can stay competitive on parts of the broader suite, but [RMSProp](REFERENCES.md#rmsprop) and [SGD+momentum](REFERENCES.md#sgd-and-momentum) still remain stronger on many standard tasks. The repository keeps that limitation visible on purpose.

## Reports

Main public report paths:

- [reports/accepted_bcdo](reports/accepted_bcdo)
- [reports/bcdo_mainline](reports/bcdo_mainline)
- [reports/reference_cnn_branch](reports/reference_cnn_branch)
- [reports/bcdo_cnn_probe](reports/bcdo_cnn_probe)
- [reports/bcdo_pinn_probe](reports/bcdo_pinn_probe)
- [reports/bcdo_mps_probe](reports/bcdo_mps_probe)
- [reports/repo_readiness_audit.md](reports/repo_readiness_audit.md)

Historical exploratory material is kept out of the accepted public line and should not be read as the first-release benchmark story.

The accepted public reports now point back to [REFERENCES.md](REFERENCES.md) whenever they discuss external optimizer families. That keeps the report layer grounded in one checked bibliography instead of a pile of partial in-file lists.

## Limitations

- BCDO is not a universal optimizer.
- It is still slower than [SGD](REFERENCES.md#sgd-and-momentum), [RMSProp](REFERENCES.md#rmsprop), and [AdamW](REFERENCES.md#adam-and-adamw).
- CNN behavior is still under development.
- PINNs are not currently a winning target for this optimizer family.
- The best current interpretation is a structured specialist/generalist hybrid, not a broad default replacement.

If you want the longer explanation of those limits, including why the reference CNN branch is still separate, see [docs/CNN_REFERENCE.md](docs/CNN_REFERENCE.md) and [reports/repo_update_report.md](reports/repo_update_report.md).

## Roadmap

- improve CNN and filter-level consensus without destabilizing dense tasks
- reduce runtime overhead in the block scoring hot path
- test larger GPU vision tasks
- compare against stronger [Muon](REFERENCES.md#muon), [Shampoo](REFERENCES.md#shampoo), and [SAM](REFERENCES.md#sam-and-asam) baselines
- refine matrix-consensus behavior on low-rank and grouped-structure tasks

## Reproducibility

See [REPRODUCING.md](REPRODUCING.md) for environment setup, focused tests, smoke runs, benchmark commands, expected output locations, and the accepted result source.

## Citation

```bibtex
@software{reid_bcdo_2026,
  title = {Blockwise Consensus Direction Optimizer},
  author = {Steven Reid},
  year = {2026},
  note = {Use the accepted benchmark snapshot and internal class name when reporting exact results.}
}
```
