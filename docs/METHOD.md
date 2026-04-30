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

## Why the accepted mainline uses the fast internal path

- cleaner than later branch experiments
- better dense/stress balance than the reference CNN branch
- preserves the strongest specialist wins
- easier to explain and maintain

There is also a reproducibility reason. The accepted public benchmark snapshot, the public wrapper scripts, and the public import alias all now resolve back to one implementation path. That makes it much easier for an external reader to connect the README, the code, and the accepted report numbers.

## Why the reference CNN branch remains

`BCDOCNNReference` is not dead code. It remains the reference branch for the CNN-trust hypothesis and still performs better than the mainline on some CNN-oriented cases. The public explanation of that branch is kept separately in [CNN_REFERENCE.md](CNN_REFERENCE.md) so that the accepted BCDO line can stay clean without pretending the branch history never happened.
