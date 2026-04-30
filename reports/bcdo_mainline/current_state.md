# Blockwise Consensus Direction Optimizer Current State

The public BCDO mainline is implemented internally as `BlockDirectionOptimizerV4Fast` and keeps the legacy fast path and the reference CNN branch as explicit comparison baselines.

## Why the accepted BCDO mainline exists

- Earlier block branches showed two durable strengths: the fast internal path had the best dense/stress speed-quality balance, while the reference CNN branch had the best CNN/block behavior.
- Heavier descendant additions did not all pay for themselves.
- The pieces that kept validating were simpler: block structure, trusted direction memory, stable consensus, typed conv/dense profiles, and conv-safe step control.

## What BCDO keeps in the accepted mainline

- Keeps only four default candidates: `gradient`, `stable_consensus`, `trusted_direction`, and `low_rank_matrix`.
- Uses `winner_take_all` by default.
- Removes per-candidate recoverability from the default hot path.
- Uses typed conv/dense profiles inside the same optimizer class.
- Uses cheap conv structure support and conv-safe scaling instead of the heavier reference-branch trust path.
- Keeps `smart_v4` grouping and block-energy normalization.

## What the accepted mainline is trying to prove

- Preserve the established stress-task edge.
- Absorb useful CNN/general-task behavior from the reference branch without taking on its full runtime and logic cost.
- Keep one main block optimizer line instead of multiple near-duplicate descendants.
