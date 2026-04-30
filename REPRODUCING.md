# Reproducing BCDO Results

This repository is set up so a fresh clone can run focused validation and reproduce the accepted Blockwise Consensus Direction Optimizer report flow without relying on unrelated workspace code.

If you need the method explanation and the external optimizer references alongside the commands below, use [docs/CLAIM.md](docs/CLAIM.md), [docs/METHOD.md](docs/METHOD.md), [docs/COMPARISONS.md](docs/COMPARISONS.md), and [REFERENCES.md](REFERENCES.md).

If you are new to the repo, the shortest useful path is:

1. read [docs/METHOD.md](docs/METHOD.md)
2. run the smoke command once
3. read [reports/accepted_bcdo/final_report.md](reports/accepted_bcdo/final_report.md)

That gives you the public method, the public code path, and the accepted benchmark line before you dig into the reference CNN branch or the probe folders.

If you are reviewing the repo as a draft paper package rather than only as code, the shortest useful follow-up path is:

1. run `python scripts/build_paper_artifacts.py`
2. read `paper/bcdo_claims_audit.md`
3. read `paper/bcdo_draft.md`
4. inspect `paper/tables/` and `paper/figures/`

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Focused tests

Run only the repo-local tests used to validate the public BCDO surface and benchmark outputs:

```bash
pytest tests/test_bcdo.py -q
pytest tests/test_blockwise_consensus_direction_optimizer.py tests/test_bcdo_benchmark_outputs.py -q
```

## Minimal examples

```bash
python examples/bcdo_minimal_mlp.py
python examples/bcdo_blockwise_direction_demo.py
python examples/bcdo_compare_against_adamw.py
```

## Smoke run

```bash
python scripts/run_bcdo_smoke.py
```

Public wrapper:

```bash
python scripts/run_bcdo_smoke.py
```

Expected output location:

- `reports/bcdo_mainline/smoke_results.csv`

## Benchmarks

```bash
python scripts/run_bcdo_tuning.py --config configs/bcdo_tuning.yaml
python scripts/run_bcdo_benchmarks.py --config configs/bcdo_default.yaml
python scripts/run_bcdo_ablation.py --config configs/bcdo_ablation.yaml
```

Expected output location:

- `reports/bcdo_mainline/benchmark_results.csv`
- `reports/bcdo_mainline/tuning_results.csv`
- `reports/bcdo_mainline/ablation_results.csv`
- `reports/bcdo_mainline/final_report.md`

## Accepted result source

The accepted first-release BCDO snapshot is stored separately from runnable outputs:

- `reports/accepted_bcdo/benchmark_results.csv`
- `reports/accepted_bcdo/final_report.md`

These accepted artifacts are the source for the README result summary and the repo readiness audit.

They are kept separate from `reports/bcdo_mainline/` on purpose. The accepted snapshot is the public first-release benchmark line. The runnable mainline folder is the current writable output path for fresh local script runs.

## Notes

- Use isolated serial runs. Do not overlap tuning, benchmark, and ablation jobs if you care about interpretable runtime data.
- CPU and MPS paths are supported in this repo. CUDA tests run conditionally when CUDA is available.
- Exploratory descendant reports are not part of the accepted public benchmark line.
- Failure boundaries and weaker domains are documented in [docs/FAILURE_CASES.md](docs/FAILURE_CASES.md).

## Paper artifacts

```bash
python scripts/build_paper_artifacts.py
python scripts/run_paper_smoke.py
```

Expected paper output locations:

- `paper/bcdo_draft.md`
- `paper/bcdo_claims_audit.md`
- `paper/bcdo_results_summary.csv`
- `paper/tables/`
- `paper/figures/`

The paper build script reads only the checked-in local CSVs and probe artifacts. It does not invent missing results or rerun the full benchmark suite automatically.
