# BCDO Method Notes

This note explains the accepted public mainline behind the Blockwise Consensus Direction Optimizer. External comparison families mentioned here are indexed in [../REFERENCES.md](../REFERENCES.md), and the broader side-by-side interpretation lives in [COMPARISONS.md](COMPARISONS.md).

## Main idea

The block optimizer family is built around a single design decision:

**choose a direction at the block level, then scale it conservatively.**

That differs from optimizers whose main mechanism is transforming one gradient direction with momentum, rescaling that direction coordinate-wise, or perturbing the objective before recomputing the gradient. BCDO treats direction choice as the first problem and step scaling as the second problem.

The accepted implementation still carries the internal name `BlockwiseConsensusDirectionOptimizer`, but that name is only a compatibility hook. The public method name for this repository is BCDO, and the public import alias is `BlockwiseConsensusDirectionOptimizer`.

## Candidate set

The accepted BCDO mainline, implemented internally as `BlockwiseConsensusDirectionOptimizer`, keeps the hot path intentionally small:

- `gradient`
- `stable_consensus`
- `trusted_direction`
- `low_rank_matrix`

Those candidates are meant to answer four different questions. The raw gradient asks what the immediate local descent direction is. The trusted-direction memory asks whether the model has been following a direction that still seems coherent. The stable-consensus candidate asks whether several signals agree strongly enough to combine. The low-rank matrix candidate asks whether a structured two-dimensional tensor offers a better row/column direction than the raw flattened block would suggest.

The optimizer family explored richer candidate sets in earlier branches, but those additions did not consistently justify their runtime cost. The accepted mainline keeps the set intentionally small because the branch only remains interesting if its blockwise logic survives a practicality check.

## Why this is not just a renamed adaptive optimizer

BCDO does keep running state, and it does use that state in its score. What it does not do is let a first-moment or second-moment recursion define the whole update direction the way [Adam or AdamW](../REFERENCES.md#adam-and-adamw) do. The block energy term in the accepted mainline controls step magnitude after a block direction has already been selected.

That separation matters. If the direction were just an Adam-like direction with one more gate or one more correction factor, then BCDO would be a renamed adaptive optimizer. The accepted mainline is trying to preserve a genuinely different local rule: discrete candidate selection before blockwise scaling.

## Structure handling

The accepted mainline uses structure in two places:

1. `low_rank_matrix` candidate for matrix-shaped blocks
2. conv-safe scaling for convolutional tensors

Convolutional support is used as a **safety and scaling** signal, not as a separate direction generator in the mainline. That design was chosen because the heavier CNN branch did show useful ideas, but those ideas did not cleanly justify becoming the default public line.

## What the trust score means in practice

The trust score is not one mysterious number. It is a way of translating several ordinary optimizer concerns into a block-level selection rule.

- Descent alignment asks whether the candidate still behaves like useful descent for the current block.
- Memory coherence asks whether the candidate agrees with directions that have recently remained usable instead of flipping abruptly.
- Quality memory keeps a candidate from surviving only because it is recent. It rewards directions that were recently useful rather than merely persistent.
- Norm stability penalizes candidates whose scale changes in a way that usually signals unstable local geometry or noisy block behavior.
- Consensus support rewards agreement across structured slices such as rows, columns, or convolutional support patterns.
- Oscillation and conflict penalties exist to stop the selector from following a direction that looks locally strong but is already reversing or fighting adjacent signals.
- Cost penalty is practical rather than geometric. It stops a more elaborate structured candidate from winning by default unless it earns that cost.

In other words, the trust score is trying to answer a practical optimizer question: if several directions are plausible, which one has the best mix of immediate descent, recent reliability, structural support, and acceptable complexity?

## What is empirically supported and what remains heuristic

The public line should not be described as though every term in the score was derived from theory and then independently verified. The exact score composition and weighting remain heuristic. What the repo does have is an accepted ablation filter that tells you which pieces survived contact with the benchmark suite strongly enough to stay in the public line.

From [../reports/accepted_bcdo/final_report.md](../reports/accepted_bcdo/final_report.md), the accepted ablation deltas were:

- block structure versus tensor collapse: `+0.0272`
- matrix consensus: `+0.0222`
- typed conv/dense split: `+0.0172`
- cheap conv structure support: `+0.0139`
- stable consensus: `+0.0137`
- trusted-direction memory: `+0.0089`
- recoverability periodic gate: `-0.0130`
- conv support bonus: `0.0000`
- conv fallback relaxation: `0.0000`

That does not turn the score into a theorem. It does justify the current public design choices. The accepted line keeps the parts that helped repeatedly enough to survive the suite, drops the recoverability gate from the default path because it hurt, and does not promote the neutral conv bonus and fallback extras into the public mainline.

## What the accepted mainline does and does not prove

The accepted line proves that a small candidate set, winner-take-all selection, typed block handling, and bounded block-energy scaling can produce a coherent and reproducible benchmark story. It does not prove that the exact trust-score decomposition is unique, optimal, or theoretically complete. It also does not prove that every structured candidate should be promoted into the public default. The reference CNN branch remains in the repo precisely because some conv-aware ideas were useful without becoming clean first-release mainline choices.

## Why the accepted mainline uses the fast internal path

- cleaner than later branch experiments
- better dense/stress balance than the reference CNN branch
- preserves the strongest specialist wins
- easier to explain and maintain

There is also a reproducibility reason. The accepted public benchmark snapshot, the public wrapper scripts, and the public import alias all now resolve back to one implementation path. That makes it much easier for an external reader to connect the README, the code, and the accepted report numbers.

## Why the reference CNN branch remains

`BCDOCNNReference` is not dead code. It remains the reference branch for the CNN-trust hypothesis and still performs better than the mainline on some CNN-oriented cases. The public explanation of that branch is kept separately in [CNN_REFERENCE.md](CNN_REFERENCE.md) so that the accepted BCDO line can stay clean without pretending the branch history never happened.
