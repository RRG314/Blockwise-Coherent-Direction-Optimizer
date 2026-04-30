# Contributing

Thanks for working on the Blockwise Consensus Direction Optimizer repository.

## Scope

This repository is intentionally narrow:

- public optimizer: `BlockwiseConsensusDirectionOptimizer`
- internal mainline class: `BlockDirectionOptimizerV4Fast`
- reference CNN branch: `BlockDirectionOptimizerV42`
- shared research harness used to reproduce the accepted result snapshots

If you want to add another optimizer family, it should go in its own repository instead of broadening this one.

## Development workflow

1. Create or activate a virtual environment.
2. Install editable dependencies:

```bash
pip install -e .[dev]
```

3. Run branch-local tests only:

```bash
pytest tests/test_block_direction_v4_fast.py tests/test_block_direction_v42.py -q
```

4. Run benchmark stages serially, not in parallel:

```bash
python scripts/run_block_direction_v4_fast_smoke.py
python scripts/run_block_direction_v4_fast_tuning.py --config configs/block_direction_v4_fast_tuning.yaml
python scripts/run_block_direction_v4_fast_benchmarks.py --config configs/block_direction_v4_fast_default.yaml
python scripts/run_block_direction_v4_fast_ablation.py --config configs/block_direction_v4_fast_ablation.yaml
python scripts/export_block_direction_v4_fast_report.py
```

Use the matching `v42` scripts for the reference CNN branch.

## Rules

- Do not weaken baselines.
- Do not cite interrupted or mixed-load runs.
- Keep the BCDO mainline on the accepted `BlockDirectionOptimizerV4Fast` path unless a replacement wins on both quality and practicality.
- Keep `BlockDirectionOptimizerV42` as a reference branch unless its conv-specific ideas are cleanly absorbed into the mainline.
- Prefer isolated, serial benchmark runs so runtime data stays interpretable.

## Pull request expectations

- Explain which branch changed: the BCDO mainline or the reference CNN branch
- Include task families affected
- Include any benchmark deltas and whether they are accepted or experimental
- Call out regressions plainly
