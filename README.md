# Blockwise Consensus Direction Optimizer

A PyTorch optimizer that selects update directions at the block level instead of transforming a single gradient direction.

`BlockwiseConsensusDirectionOptimizer` is the public class for the accepted mainline in this repository. The optimizer is not presented here as a universal replacement for `SGD`, `RMSProp`, `AdamW`, or more structured preconditioning methods. Its intended use is narrower: training problems where the raw gradient direction is unstable, conflicting, structurally misleading, or too myopic to be trusted on its own.

The current evidence supports BCDO as a structured specialist/generalist hybrid. The accepted reports show real wins on stress-oriented and structure-sensitive slices, but they also show clear limitations: `RMSProp` and `SGD+momentum` remain stronger on many standard tasks, runtime is still higher than simple baselines, and CNN behavior is improved relative to the older fast path without yet becoming a dominant public story.

## Project status

This repository is set up as a clone-and-run research package. It includes the accepted BCDO implementation, a reference CNN branch, focused tests, benchmark scripts, configuration files, exported reports, and reproducibility notes.

The public surface is:

```python
from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer
```

The current claim is intentionally narrow:

> BCDO is a blockwise direction-selection optimizer for regimes where the raw gradient direction is unstable, conflicting, or structurally misleading.

It should not be read as a claim that BCDO is the best general-purpose optimizer for ordinary supervised learning, CNN training, PINNs, or broad default PyTorch use.

## Overview

Most optimizers start from one gradient direction and then smooth it, rescale it, or precondition it. BCDO takes a different route: for each parameter block it builds a small set of candidate directions, scores them with structural trust signals, chooses the most trusted direction with winner-take-all selection, and applies a bounded blockwise step.

This is the central idea of the repository. BCDO is not trying to be “AdamW plus extra gates.” It is also not trying to become a full second-order optimizer, a curvature model, or a matrix-preconditioned natural-gradient method. The main decision is discrete and local: which direction should this block trust right now?

The public documentation is organized around that question. The method explanation is in [docs/METHOD.md](docs/METHOD.md), the comparison notes are in [docs/COMPARISONS.md](docs/COMPARISONS.md), the literature used in those comparisons is collected in [REFERENCES.md](REFERENCES.md), and the attribution note for repository preparation is in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

The public optimizer name in this repository is `BlockwiseConsensusDirectionOptimizer`, with `BCDO` as the short alias used in code and reports.

The repository also keeps one reference CNN branch:

- `BCDOCNNReference` (documented in [docs/CNN_REFERENCE.md](docs/CNN_REFERENCE.md))

## Relation to existing optimizers

BCDO is easiest to understand by contrast with the optimizer families it is measured against. `SGD` follows the raw gradient directly and `SGD+momentum` replaces that raw direction with one smoothed velocity recursion; see [SGD and momentum](REFERENCES.md#sgd-and-momentum). `RMSProp` keeps the basic gradient direction but rescales it coordinate-wise with squared-gradient history; see [RMSProp](REFERENCES.md#rmsprop). `Adam` and `AdamW` combine momentum and adaptive scaling in a single per-parameter update; see [Adam and AdamW](REFERENCES.md#adam-and-adamw). `Muon`, `Shampoo`, and `K-FAC` use matrix or tensor structure more aggressively, but they do so through orthogonalization or preconditioning rather than through explicit candidate-direction choice; see [Muon](REFERENCES.md#muon), [Shampoo](REFERENCES.md#shampoo), and [K-FAC](REFERENCES.md#k-fac). `SAM` and `ASAM` are useful comparison points for stability and generalization, but they perturb the objective neighborhood rather than selecting among blockwise direction candidates; see [SAM and ASAM](REFERENCES.md#sam-and-asam).

BCDO is different in a narrower and more concrete way. It does not start from one gradient direction and then keep modifying that direction. It forms a small candidate set per block, scores the candidates, selects one winner, and only then decides how far to move. That is why this repository treats BCDO as a direction-selection optimizer rather than as another Adam variant with extra controls.

That distinction matters because the comparison burden is different. If BCDO were just another moment-based optimizer, the main question would be whether its controller stack beats AdamW. But BCDO is closer in spirit to a local decision rule: it uses block structure to ask whether the plain gradient, a more stable consensus direction, a trusted memory direction, or a low-rank matrix-style direction should be followed for this block at this step.

This repo also compares against a small internal coherent-momentum comparator family because those runs were part of the accepted benchmark story. Those internal comparators are treated as benchmark context, not as the public identity of this repository.

## What problem this targets

BCDO is aimed at training situations where the local gradient is not obviously the direction that should be followed block-for-block. That can happen when optimization repeatedly oscillates along narrow valleys, when block structure matters more than coordinate-wise scaling, when direction reverses under unstable local geometry, or when structured consensus across rows, filters, or tensor slices matters more than one-step raw descent.

The optimizer therefore asks a different question from standard adaptive methods. Instead of only asking how large an update should be, it also asks whether the block should trust the raw gradient, a consensus direction, a memory direction, or a structure-aware matrix direction. The accepted benchmark suite is built around that narrower question rather than around a broad “best default optimizer” claim.

For a newcomer, the most useful first pass is usually: read [docs/METHOD.md](docs/METHOD.md), run the smoke command once, and then inspect [reports/accepted_bcdo/final_report.md](reports/accepted_bcdo/final_report.md). That path shows the public code, the public claim, and the accepted result snapshot before the broader branch history or probe folders get in the way.

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

In practical optimizer terms, these terms do different jobs. Descent alignment asks whether a candidate still points downhill for the current block. Memory coherence and quality memory ask whether it agrees with directions that were recently useful, instead of only being large or recent. Norm stability penalizes abrupt scale changes that often accompany unstable local geometry. Consensus support rewards candidates that line up with structured agreement across a matrix or convolutional block. Oscillation and conflict penalties try to keep the selector from following a direction that looks locally plausible but is already reversing or fighting nearby signals. The cost term is there to keep more structured candidates from winning unless they actually earn that added complexity.

### Selection rule

BCDO uses winner-take-all selection:

`d_i* = argmax_d T_i(d)`

### Step rule

The selected block direction is scaled with a bounded block-energy rule:

`alpha_i = lr * trust_i * ||g_i|| / (sqrt(E_i) + eps)^p`

Convolutional tensors use typed conv-safe scaling on top of the same direction-selection core. The accepted mainline deliberately keeps that convolutional logic on the scaling side rather than turning it into another full candidate generator, because earlier branch work did not justify the heavier alternative on the accepted benchmark line.

The important practical point is that BCDO keeps direction choice and magnitude control separate. Standard adaptive methods usually intertwine those operations. BCDO first decides which candidate direction to trust and then scales the chosen direction with a bounded block-energy rule. That separation is what makes the method easier to inspect and easier to compare honestly against both first-order baselines and more structured matrix-aware baselines.

The repo does not claim that every part of that score is theory-derived. The candidate-selection structure, winner-take-all rule, and block-energy step are the central design choices. The exact score balance remains heuristic and is only kept in the public line where the accepted ablation results support it.

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

The public import is:

```python
from optimizers.blockwise_consensus_direction_optimizer import BlockwiseConsensusDirectionOptimizer
```

Backward-compatible imports still work:

```python
from optimizers import BCDO, BlockwiseConsensusDirectionOptimizer
```

The reference CNN branch remains importable as `BCDOCNNReference`, but it is not the accepted first-release mainline.

## Installation

This repository is installable with the included `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

The default dependency set covers the included smoke, tuning, benchmark, and export scripts. The current public benchmark story is CPU-oriented, with MPS probe support included separately in the reports.

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

## Running tests

Focused repo-local tests:

```bash
pytest tests/test_bcdo.py -q
pytest tests/test_blockwise_consensus_direction_optimizer.py tests/test_bcdo_benchmark_outputs.py -q
```

These are the tests used as repo-readiness evidence. They cover import paths, initialization, one-step parameter updates, no-NaN smoke behavior, diagnostics, candidate-selection logic, winner-take-all behavior, convolutional tensor handling, matrix-consensus handling, and benchmark-output schemas.

Use these focused tests for this repository. Full parent-workspace failures are not relevant readiness evidence for BCDO.

If you only want one command to check that the public surface is alive before you read deeper, run `python scripts/run_bcdo_smoke.py`.

## Running benchmarks

Mainline scripts:

```bash
python scripts/run_bcdo_smoke.py
python scripts/run_bcdo_tuning.py --config configs/bcdo_tuning.yaml
python scripts/run_bcdo_benchmarks.py --config configs/bcdo_default.yaml
python scripts/run_bcdo_ablation.py --config configs/bcdo_ablation.yaml
python scripts/export_bcdo_report.py
```

Reference CNN branch scripts:

```bash
python scripts/run_bcdo_cnn_reference_smoke.py
python scripts/run_bcdo_cnn_reference_tuning.py --config configs/bcdo_cnn_reference_tuning.yaml
python scripts/run_bcdo_cnn_reference_benchmarks.py --config configs/bcdo_cnn_reference_default.yaml
python scripts/run_bcdo_cnn_reference_ablation.py --config configs/bcdo_cnn_reference_ablation.yaml
python scripts/export_bcdo_cnn_reference_report.py
```

The accepted public snapshot and the current runnable mainline are kept separate on purpose. That lets the repository preserve the accepted first-release benchmark line while still supporting new smoke and export runs from the current code.

That split is part of the repo’s public discipline, not a packaging accident. The accepted snapshot is the line the README cites. The runnable mainline is where fresh local runs should write. Keeping them separate avoids quietly replacing the first-release benchmark story every time the scripts are re-run.

## What it compares against

The accepted mainline benchmark config keeps the baselines that matter most for the BCDO claim:

- `AdamW`
- `RMSProp`
- `SGD+momentum`
- the legacy fast path
- the reference CNN branch
- the internal coherent momentum comparator family

The goal of that comparison set is not to create a maximal optimizer zoo. It is to keep the comparison burden honest around the actual question this repository is asking: whether blockwise direction selection improves the update decision in structure-sensitive or stress-sensitive regimes without collapsing back into a standard first-order controller.

Methods such as `Muon`, `Shampoo`, `K-FAC`, and `SAM` still appear in the docs because they are the right literature context for a structure-aware optimizer. They are not presented as if they were part of the accepted first-release leaderboard when the checked-in public reports did not run that direct comparison.

## Results

The repository contains several report groups that matter for orientation:

1. the accepted public mainline snapshot in [reports/accepted_bcdo](reports/accepted_bcdo)
2. the current runnable mainline in [reports/bcdo_mainline](reports/bcdo_mainline)
3. the reference CNN branch in [reports/reference_cnn_branch](reports/reference_cnn_branch)
4. focused probes in [reports/bcdo_cnn_probe](reports/bcdo_cnn_probe), [reports/bcdo_pinn_probe](reports/bcdo_pinn_probe), and [reports/bcdo_mps_probe](reports/bcdo_mps_probe)

### Accepted mainline snapshot

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

Those results support a narrow claim, not a broad one. BCDO has a real edge on the accepted stress-oriented slice and stays competitive on parts of the broader suite, but [RMSProp](REFERENCES.md#rmsprop) and [SGD+momentum](REFERENCES.md#sgd-and-momentum) still remain stronger on many standard tasks.

### Current interpretation

The evidence supports a structured, narrower conclusion. BCDO is most defensible when the problem is not simply “adaptive scaling versus momentum,” but rather “which blockwise direction should be trusted at all?” That is where the candidate-direction design matters.

The current evidence does **not** support a universal optimizer claim, a broad CNN leadership claim, or a replacement claim over `RMSProp` or `SGD+momentum`. The repo keeps those limits visible on purpose because they are part of what makes the accepted benchmark story credible.

A safe public sentence to repeat from the current repo is: BCDO is a blockwise direction-selection optimizer that can help on structure-sensitive and stress-sensitive slices, but it is not a general replacement for RMSProp, SGD with momentum, or AdamW.

### Reference CNN branch and probe summary

The repository keeps a separate reference CNN branch because CNN-side behavior improved through typed conv-safe scaling and conv-aware trust rules, but those ideas were not fully absorbed into the accepted public mainline without tradeoffs. The reference CNN branch and the focused CNN probe reports are included so that a newcomer can see both what helped and what still remains unresolved.

### PINN and MPS probes

The PINN and MPS probe folders are included for completeness, but they are not the core public claim for BCDO. The repository does not present BCDO as a winning PINN optimizer, and it does not use MPS probe runs as the basis for broad performance claims.

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

BCDO is not a universal optimizer. It is still slower than [SGD](REFERENCES.md#sgd-and-momentum), [RMSProp](REFERENCES.md#rmsprop), and [AdamW](REFERENCES.md#adam-and-adamw). CNN behavior is still under development, and PINNs are not currently a winning target for this optimizer family. The best current interpretation is a structured specialist/generalist hybrid, not a broad default replacement.

If you want the longer explanation of those limits, including why the reference CNN branch is still separate, see [docs/CNN_REFERENCE.md](docs/CNN_REFERENCE.md) and [reports/repo_update_report.md](reports/repo_update_report.md).

The fuller failure-boundary note is in [docs/FAILURE_CASES.md](docs/FAILURE_CASES.md).

## Known failure cases

Short version: on clean standard tasks, `RMSProp` or `SGD+momentum` may be the better choice. On CNNs, the mainline BCDO path still trails the strongest practical baselines in this repository. On PINN-style problems, the included probe results do not justify a strong public claim. On stable smooth problems where the raw gradient direction is already usable, simpler optimizers are cheaper and often sufficient.

## Roadmap

Current practical next steps are to improve CNN and filter-level consensus without destabilizing dense tasks, reduce runtime overhead in the block scoring hot path, test larger GPU vision tasks, compare against stronger [Muon](REFERENCES.md#muon), [Shampoo](REFERENCES.md#shampoo), and [SAM](REFERENCES.md#sam-and-asam) baselines, and refine matrix-consensus behavior on low-rank and grouped-structure tasks.

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
