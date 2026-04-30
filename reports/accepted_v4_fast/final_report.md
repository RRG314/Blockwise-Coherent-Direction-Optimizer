# BlockDirectionOptimizerV4Fast Final Report

## 1. What V4Fast is

- V4Fast is now the consolidated main block branch.
- It keeps the novel blockwise direction-selection principle, preserves the fast dense/stress core, and folds in the lowest-cost validated CNN/general-task elements from V4.2.

## 2. How merged V4Fast differs from the older fast path

- Candidate set stays reduced to gradient, stable consensus, trusted direction, and matrix consensus.
- Winner-take-all selection stays the default.
- Recoverability stays out of the default hot path.
- Dense/vector tensors use the fast original profile.
- Convolutional tensors use a typed conv profile, cheap structure-aware conv-safe scaling, and a low-cost conv-aware step rule.

## 3. Best rows

- Best V4Fast row: `breast_cancer_mlp` with mean best val loss `0.061726` and mean best val accuracy `0.985677`.
- Best legacy V4Fast row: `breast_cancer_mlp` with mean best val loss `0.061726` and mean best val accuracy `0.985677`.
- Best V4.2 row: `wine_mlp` with mean best val loss `0.052929` and mean best val accuracy `0.985185`.
- Strongest baseline row: `sgd_momentum` on `wine_mlp` with mean best val loss `0.009711` and mean best val accuracy `1.000000`.

## 4. Competitive summary

- V4Fast wins vs legacy V4Fast: `4` meaningful wins.
- V4Fast wins vs V4.2: `4` meaningful wins.
- V4Fast wins vs AdamW: `7` meaningful wins; tracked 2x wins `5`.
- V4Fast wins vs RMSProp: `6` meaningful wins.
- V4Fast wins vs SGD momentum: `4` meaningful wins.
- V4Fast wins vs MagnetoHamiltonianAdam: `10` meaningful wins.

## 5. Runtime

- Mean V4Fast runtime per step: `39.5980 ms`.
- Mean legacy V4Fast runtime per step: `41.5548 ms`.
- Mean V4.2 runtime per step: `32.9175 ms`.

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

- Best task family for V4Fast: `stability`.
- The merged V4Fast only counts as a success if it keeps the old stress-task strengths and improves the broader task mix relative to legacy V4Fast and V4.2.
- If RMSProp or SGD momentum still dominate broadly, V4Fast remains a specialist optimizer rather than a new default.
