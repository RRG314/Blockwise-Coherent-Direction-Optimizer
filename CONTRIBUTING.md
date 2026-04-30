# Contributing

Thanks for working on the Blockwise Consensus Direction Optimizer repository.

## Scope

This repository is intentionally narrow:

- public optimizer: `BlockwiseConsensusDirectionOptimizer`
- reference CNN branch: `BCDOCNNReference`
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
pytest tests/test_bcdo.py tests/test_bcdo_cnn_reference.py -q
```

4. Run benchmark stages serially, not in parallel:

```bash
python scripts/run_bcdo_smoke.py
python scripts/run_bcdo_tuning.py --config configs/bcdo_tuning.yaml
python scripts/run_bcdo_benchmarks.py --config configs/bcdo_default.yaml
python scripts/run_bcdo_ablation.py --config configs/bcdo_ablation.yaml
python scripts/export_bcdo_report.py
```

Use the matching `bcdo_cnn_reference` scripts for the reference CNN branch when you are reproducing that older comparison path.

## Rules

- Do not weaken baselines.
- Do not cite interrupted or mixed-load runs.
- Keep the BCDO mainline on the accepted `BlockwiseConsensusDirectionOptimizer` path unless a replacement wins on both quality and practicality.
- Keep `BCDOCNNReference` as a reference branch unless its conv-specific ideas are cleanly absorbed into the mainline.
- Prefer isolated, serial benchmark runs so runtime data stays interpretable.

## Pull request expectations

- Explain which branch changed: the BCDO mainline or the reference CNN branch
- Include task families affected
- Include any benchmark deltas and whether they are accepted or experimental
- Call out regressions plainly
