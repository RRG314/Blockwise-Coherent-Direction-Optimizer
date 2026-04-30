# BlockDirectionOptimizerV4.2 Final Report

## 1. What V4.2 is

- V4.2 is a conv-aware refinement of V4Fast that uses conv structure as trust, not as a default new direction.
- It keeps the V4Fast candidate set unchanged and uses conv-safe scaling plus conv-aware trust/fallback rules.

## 2. Best rows

- Best V4.2 row: `breast_cancer_mlp` with mean best val loss `0.058269` and mean best val accuracy `0.988281`.
- Best V4Fast row on the same suite: `breast_cancer_mlp` with mean best val loss `0.059664` and mean best val accuracy `0.986979`.
- Strongest baseline row: `rmsprop` on `breast_cancer_mlp` with mean best val loss `0.056732` and mean best val accuracy `0.988281`.

## 3. Competitive summary

- V4.2 wins vs V4Fast: `4` meaningful wins.
- V4.2 wins vs V4.1: `5` meaningful wins.
- V4.2 wins vs AdamW: `4` meaningful wins.
- V4.2 wins vs RMSProp: `3` meaningful wins.
- V4.2 wins vs SGD momentum: `5` meaningful wins.
- V4.2 wins vs MagnetoHamiltonianAdam: `6` meaningful wins.

## 4. Ablation findings

- Conv structure bonus delta vs removal: `-0.0006`.
- Conv fallback relaxation delta vs removal: `0.0000`.
- Conv safe scaling delta vs removal: `0.0102`.
- Best ablation row overall: `rmsprop_baseline`.

## 5. Honest conclusion

- V4.2 only succeeds if it improves standard/CNN behavior without giving back V4Fast's stress-task wins.
- If RMSProp still wins CNN tasks cleanly, V4.2 remains a specialist rather than a broadly usable default.
