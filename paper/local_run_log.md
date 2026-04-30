# Local Run Log

This log records the local verification commands run for the paper-package hardening pass on 2026-04-30.

## Environment

- machine: local workstation
- execution mode: local only
- remote benchmark services: not used
- external APIs: not used

## Commands run

### Focused tests

```bash
../../.venv_research/bin/pytest tests/test_bcdo.py -q
```

Result:

- `7 passed in 108.50s`

```bash
../../.venv_research/bin/pytest tests/test_blockwise_consensus_direction_optimizer.py tests/test_bcdo_benchmark_outputs.py -q
```

Result:

- `4 passed in 1.76s`

### Examples

```bash
../../.venv_research/bin/python examples/bcdo_minimal_mlp.py
```

Result:

- best validation loss: `0.039748`
- best validation accuracy: `1.000000`

```bash
../../.venv_research/bin/python examples/bcdo_blockwise_direction_demo.py
```

Result:

- best validation loss: `0.227013`
- final validation loss: `0.227013`
- selected candidate type: `gradient`
- trust score: `0.9978042850270867`
- consensus strength: `0.9995989566668868`
- saved plot: `reports/bcdo_mainline/example_blockwise_direction_demo.png`

Note:

- this example is meant to show the blockwise diagnostic surface on a structured matrix problem
- it is not presented as a guaranteed win over AdamW

```bash
../../.venv_research/bin/python examples/bcdo_compare_against_adamw.py
```

Result:

- BCDO best validation loss: `0.082148`
- BCDO best validation accuracy: `0.972028`
- AdamW best validation loss: `0.088239`
- AdamW best validation accuracy: `0.965035`

### Paper artifacts

```bash
../../.venv_research/bin/python scripts/build_paper_artifacts.py
```

Result:

- generated `paper/bcdo_results_summary.csv`
- generated `paper/bcdo_claims_audit.md`
- generated `paper/bcdo_draft.md`
- generated `paper/tables/`
- generated `paper/figures/`

```bash
../../.venv_research/bin/python scripts/run_paper_smoke.py
```

Result:

- passed
- verified compile step, focused tests, examples, and paper artifact generation

## Commands not run

None of the required local verification commands were skipped during this pass.

## Notes

- The paper artifact build reads only checked-in local CSVs and markdown reports.
- No fresh full benchmark rerun was required for this pass.
- A documentation reconciliation pass was needed because some older runtime numbers in the accepted public prose no longer matched the checked-in benchmark CSV aggregation. The paper package now follows the CSV-derived runtime values.
