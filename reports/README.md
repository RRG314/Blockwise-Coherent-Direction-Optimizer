# Reports Inventory

This folder separates accepted public artifacts from exploratory history.

External optimizer families mentioned in the accepted public reports are documented centrally in [../REFERENCES.md](../REFERENCES.md). Internal compatibility names such as `BlockDirectionOptimizerV4Fast` and `BlockDirectionOptimizerV42` still appear in some report files because the accepted results were produced from those concrete code paths.

## Public report paths

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

## Historical paths kept for compatibility

- `accepted_v4_fast/`
- `reference_v42/`
- `cnn_probe_v4_fast/`
- `pinn_probe_v4_fast/`
- `mps_probe_v4_fast/`

These remain so older scripts, notes, and links still resolve.

## Exploratory history

- `block_direction_experimental_backup_20260429/`

This backup folder contains historical summary documents from exploratory branch work. They are retained for context, but they are not the accepted public benchmark line.
