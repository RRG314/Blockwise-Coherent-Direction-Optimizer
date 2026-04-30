# Reports Inventory

This folder separates accepted public artifacts from exploratory history.

External optimizer families mentioned in the accepted public reports are documented centrally in [../REFERENCES.md](../REFERENCES.md). Internal compatibility names such as `BlockwiseConsensusDirectionOptimizer` and `BCDOCNNReference` still appear in some report files because the accepted results were produced from those concrete code paths.

## Public report paths

- `complete_system_inventory_and_results.md`
  - all-in-one system, inventory, and results reference for the current repo
- `accepted_bcdo/`
  - accepted first-release BCDO benchmark snapshot
- `bcdo_mainline/`
  - current runnable mainline output location used by the smoke, benchmark, ablation, and export scripts
- `reference_cnn_branch/`
  - accepted reference CNN-branch snapshot
- `bcdo_cnn_probe/`
  - focused CNN probe artifacts
- `bcdo_pinn_probe/`
  - focused PINN probe artifacts
- `bcdo_mps_probe/`
  - MPS probe artifacts

## Exploratory history

- `bcdo_experimental_backup_20260429/` if present in a local working copy or pre-cleanup archive

Exploratory backup material is not part of the accepted first-release line. The public repository should be read through `accepted_bcdo/`, `bcdo_mainline/`, and the named probe folders above.
