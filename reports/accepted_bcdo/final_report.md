# Blockwise Consensus Direction Optimizer Final Report

External optimizer families and internal comparison baselines mentioned in this document are indexed in [../../REFERENCES.md](../../REFERENCES.md).

## 1. What BCDO is

- BCDO is the public name for the accepted block-direction mainline implemented internally as `BlockwiseConsensusDirectionOptimizer`.
- It keeps the blockwise direction-selection principle, preserves the fast dense/stress core, and folds in the lowest-cost validated CNN/general-task elements from the reference CNN branch.

## 2. How the accepted BCDO mainline differs from the older fast path

- Candidate set stays reduced to gradient, stable consensus, trusted direction, and matrix consensus.
- Winner-take-all selection stays the default.
- Recoverability stays out of the default hot path.
- Dense/vector tensors use the fast original profile.
- Convolutional tensors use a typed conv profile, cheap structure-aware conv-safe scaling, and a low-cost conv-aware step rule.

## 3. Best rows

- Best BCDO row: `breast_cancer_mlp` with mean best val loss `0.061726` and mean best val accuracy `0.985677`.
- Best legacy fast-path row: `breast_cancer_mlp` with mean best val loss `0.061726` and mean best val accuracy `0.985677`.
- Best reference CNN-branch row: `wine_mlp` with mean best val loss `0.052929` and mean best val accuracy `0.985185`.
- Strongest baseline row: `sgd_momentum` on `wine_mlp` with mean best val loss `0.009711` and mean best val accuracy `1.000000`.

## 4. Competitive summary

- BCDO wins vs the legacy fast path: `4` meaningful wins.
- BCDO wins vs the reference CNN branch: `4` meaningful wins.
- BCDO wins vs [AdamW](../../REFERENCES.md#adam-and-adamw): `7` meaningful wins; tracked 2x wins `5`.
- BCDO wins vs [RMSProp](../../REFERENCES.md#rmsprop): `6` meaningful wins.
- BCDO wins vs [SGD with momentum](../../REFERENCES.md#sgd-and-momentum): `4` meaningful wins.
- BCDO wins vs [the internal Coherent Momentum baseline family](../../REFERENCES.md#internal-comparison-baselines): `10` meaningful wins.

## 5. Runtime

- Mean BCDO runtime per step: `39.5980 ms`.
- Mean legacy fast-path runtime per step: `41.5548 ms`.
- Mean reference CNN-branch runtime per step: `32.9175 ms`.

## 6. Ablation findings

- Typed conv/dense split delta vs removal: `0.0172`.
- Cheap conv structure support delta vs removal: `0.0139`.
- Conv support bonus delta vs removal: `0.0000`.
- Conv fallback-relaxation delta vs removal: `0.0000`.
- Stable consensus delta vs removal: `0.0137`.
- Trusted-direction memory delta vs removal: `0.0089`.
- Matrix consensus delta vs removal: `0.0222`.
- Block structure delta vs tensor collapse: `0.0272`.
- Recoverability periodic gate delta: `-0.0130`.
- Best ablation row overall: `rmsprop_baseline`.

## 7. Honest conclusion

- Best task family for BCDO: `stability`.
- The accepted BCDO mainline only counts as a success if it keeps the old stress-task strengths and improves the broader task mix relative to the legacy fast path and the reference CNN branch.
- If [RMSProp](../../REFERENCES.md#rmsprop) or [SGD with momentum](../../REFERENCES.md#sgd-and-momentum) still dominate broadly, BCDO remains a specialist optimizer rather than a new default.
