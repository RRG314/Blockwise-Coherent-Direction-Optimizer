# BCDO PINN Probe Current State

This probe evaluates the accepted BCDO mainline implemented internally as `BlockDirectionOptimizerV4Fast`.

External optimizer families and companion methods mentioned in this document are indexed in [../../REFERENCES.md](../../REFERENCES.md).

## Why the accepted BCDO mainline exists

- Earlier block branches were competitively interesting on oscillatory, saddle, and plateau tasks, but too slow for practical standard-task use.
- The slow path came from too many candidates, too much per-candidate scoring, and recoverability work inside the hot loop.
- The parts that actually helped were simpler: block structure, trusted direction memory, stable consensus, and winner-take-all selection.

## What the accepted BCDO mainline changes

- Keeps only four default candidates: `gradient`, `stable_consensus`, `trusted_direction`, and `low_rank_matrix`.
- Uses `winner_take_all` by default.
- Removes per-candidate recoverability from the default hot path.
- Uses the internal smart block-grouping rule: tensor blocks for vectors and small matrices, row blocks only for larger matrices.
- Replaces the older broad magnitude rule with block-energy normalization.

## What this probe is trying to prove

- Preserve the stress-task edge established by the earlier block line.
- Improve standard-task behavior enough to be usable.
- Cut runtime materially without collapsing the branch back into Adam-like logic.
