# How BCDO Differs From Standard Optimizers

This repository uses the name Blockwise Consensus Direction Optimizer (BCDO) because the core algorithmic decision is a direction-selection decision, not a gradient-transform decision. That distinction is easy to blur in short README summaries, so this note makes it explicit.

Supporting references for every optimizer family mentioned here are collected in [../REFERENCES.md](../REFERENCES.md).

## The shortest accurate description

Most commonly used optimizers start with one gradient direction and then modify that direction. They may smooth it with momentum, rescale it with squared-gradient history, precondition it with matrix statistics, or perturb the loss to make the direction flatter or more robust.

BCDO does something else first. It forms a small candidate set for each parameter block, scores those candidates with block-local trust signals, chooses one winner, and then applies a bounded blockwise step. In other words, BCDO treats direction choice as the primary problem and step scaling as the secondary problem.

## What BCDO shares with SGD and momentum

BCDO is still a first-order method. It uses gradients directly, it stores simple running state, and it is designed to be cheap enough for ordinary PyTorch training loops. In that sense it belongs closer to SGD, momentum, RMSProp, and AdamW than to expensive second-order methods.

The important difference is that SGD and momentum commit to one update direction immediately. SGD uses the current gradient. Momentum uses a velocity recursion. BCDO keeps several plausible block directions alive long enough to choose among them.

## What BCDO does not borrow from AdamW

AdamW is the strongest practical contrast point for this repository because it combines direction smoothing and coordinate-wise scaling in a single familiar update. BCDO does not replace AdamW's moment estimates with slightly different moment estimates. It does not add one more gate on top of AdamW and call that a new optimizer.

The accepted BCDO mainline does use running state, but the state does not define the direction by itself. The running energy term controls magnitude only. Trusted-direction memory and stability memory help score candidate directions, but they do not force the optimizer into one Adam-like transform rule.

## What BCDO does not borrow from RMSProp

RMSProp is a strong baseline because it is simple, robust, and often hard to beat on small and noisy tasks. Its key move is per-parameter normalization using squared-gradient history.

BCDO separates that role from direction choice. The accepted mainline uses a block-energy normalization term only after it has already chosen a block direction. That makes the step control adaptive without turning the whole method into a coordinate-wise variance preconditioner.

## What BCDO does not borrow from Muon, Shampoo, or K-FAC

Muon, Shampoo, and K-FAC matter because they are real structure-aware optimizers, not toy alternatives. They use matrix or curvature structure to orthogonalize or precondition updates.

BCDO is lighter than that family on purpose. It uses matrix structure to define candidates and support signals, not to build a full structured inverse or tensor preconditioner. If BCDO helps, it helps because cheap structured direction selection sometimes captures enough of the useful geometry without paying the full cost of those methods.

## What BCDO does not borrow from SAM

SAM and ASAM change the effective optimization problem by minimizing local sharpness as well as pointwise loss. That requires an extra perturb-and-recompute pattern.

The accepted BCDO line does not perturb the objective or the parameters in that way. It stays in a standard one-pass optimizer step and uses local history, coherence, and structure signals instead.

## What BCDO does not borrow from PCGrad or CAGrad

PCGrad and CAGrad are important when there are several explicit task gradients that conflict with one another. Their design target is multi-task optimization.

BCDO can operate in settings where conflicts appear, but it does not require multiple task-specific gradients. Its scoring rule is defined blockwise and can be evaluated in ordinary single-objective training.

## Why the accepted mainline stayed simple

This repository keeps the internal implementation name `BlockDirectionOptimizerV4Fast` for compatibility, but the public method name is BCDO. The accepted mainline stayed with the fast internal path because later branches repeatedly showed the same pattern: more candidates and more controller logic increased runtime faster than they improved results.

The current public method therefore keeps:

- a small candidate set,
- winner-take-all block selection,
- typed dense and convolutional profiles,
- cheap structure-aware scaling rather than heavy structure-aware control.

That is the cleanest version of the idea that survived the internal branch search.
