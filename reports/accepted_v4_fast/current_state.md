# BlockDirectionOptimizerV4Fast Current State

V4Fast now serves as the consolidated main block branch and keeps the legacy fast path and V4.2 as explicit comparison baselines.

## Why V4Fast exists

- Earlier block branches showed two durable strengths: V4Fast had the best dense/stress speed-quality balance, while V4.2 had the best CNN/block behavior.
- The heavy V4.2/V4.3/V4.4/V4.5 additions did not all pay for themselves.
- The pieces that kept validating were simpler: block structure, trusted direction memory, stable consensus, typed conv/dense profiles, and conv-safe step control.

## What V4Fast changes

- Keeps only four default candidates: `gradient`, `stable_consensus`, `trusted_direction`, and `low_rank_matrix`.
- Uses `winner_take_all` by default.
- Removes per-candidate recoverability from the default hot path.
- Uses typed conv/dense profiles inside the same optimizer class.
- Uses cheap conv structure support and conv-safe scaling instead of the heavier V4.2 support path.
- Keeps `smart_v4` grouping and block-energy normalization.

## What V4Fast is trying to prove

- Preserve the old V4Fast stress-task edge.
- Absorb the useful CNN/general-task behavior from V4.2 without taking on the full V4.2 runtime and logic cost.
- Keep one main block optimizer line instead of multiple near-duplicate descendants.
