# BlockDirectionOptimizerV4Fast Current State

V4Fast keeps BlockDirectionOptimizerV3 intact and adds a separate fast branch.

## Why V4Fast exists

- V3 was competitively interesting on oscillatory, saddle, and plateau tasks, but too slow for practical standard-task use.
- The slow path came from too many candidates, too much per-candidate scoring, and recoverability work inside the hot loop.
- The parts that actually helped were simpler: block structure, trusted direction memory, stable consensus, and winner-take-all selection.

## What V4Fast changes

- Keeps only four default candidates: `gradient`, `stable_consensus`, `trusted_direction`, and `low_rank_matrix`.
- Uses `winner_take_all` by default.
- Removes per-candidate recoverability from the default hot path.
- Uses `smart_v4` grouping: tensor blocks for vectors and small matrices, row blocks only for larger matrices.
- Replaces the older broad magnitude rule with block-energy normalization.

## What V4Fast is trying to prove

- Preserve V3's stress-task edge.
- Improve standard-task behavior enough to be usable.
- Cut runtime materially without collapsing the branch back into Adam-like logic.
