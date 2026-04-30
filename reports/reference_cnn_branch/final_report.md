# Reference CNN Branch Final Report

External optimizer families and internal comparison baselines mentioned in this document are indexed in [../../REFERENCES.md](../../REFERENCES.md).

## 1. What the reference CNN branch is

- The reference CNN branch is implemented internally as `BCDOCNNReference`.
- It is a conv-aware refinement of the accepted fast mainline that uses conv structure as trust, not as a default new direction.
- It keeps the accepted BCDO candidate set unchanged and uses conv-safe scaling plus conv-aware trust/fallback rules.

## 2. Best rows

- Best reference-branch row: `breast_cancer_mlp` with mean best val loss `0.058269` and mean best val accuracy `0.988281`.
- Best accepted-mainline row on the same suite: `breast_cancer_mlp` with mean best val loss `0.059664` and mean best val accuracy `0.986979`.
- Strongest baseline row: `rmsprop` on `breast_cancer_mlp` with mean best val loss `0.056732` and mean best val accuracy `0.988281`.

## 3. Competitive summary

- Reference branch wins vs the accepted mainline: `4` meaningful wins.
- Reference branch wins vs the earlier conv-safe branch: `5` meaningful wins.
- Reference branch wins vs [AdamW](../../REFERENCES.md#adam-and-adamw): `4` meaningful wins.
- Reference branch wins vs [RMSProp](../../REFERENCES.md#rmsprop): `3` meaningful wins.
- Reference branch wins vs [SGD with momentum](../../REFERENCES.md#sgd-and-momentum): `5` meaningful wins.
- Reference branch wins vs [the internal Coherent Momentum baseline family](../../REFERENCES.md#internal-comparison-baselines): `6` meaningful wins.

## 4. Ablation findings

- Conv structure bonus delta vs removal: `-0.0006`.
- Conv fallback relaxation delta vs removal: `0.0000`.
- Conv safe scaling delta vs removal: `0.0102`.
- Best ablation row overall: `rmsprop_baseline`.

## 5. Honest conclusion

- This branch only succeeds if it improves standard/CNN behavior without giving back the accepted mainline's stress-task wins.
- If [RMSProp](../../REFERENCES.md#rmsprop) still wins CNN tasks cleanly, the branch remains a specialist rather than a broadly usable default.
