# Focused Claim

References for the baseline families named here are collected in [../REFERENCES.md](../REFERENCES.md).

The narrow public claim for this repository is:

**BCDO is a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, structurally misleading, or too myopic to trust block-for-block.**

This repository uses that claim in a specific sense.

- The method is aimed at situations where block structure matters and the immediate gradient is not always the most trustworthy block-level direction.
- The accepted first-release line is strongest on stress-oriented and structure-sensitive slices.
- The public story depends on blockwise candidate selection and conservative block-energy scaling, not on broad claims about ordinary default optimization.

## What the current evidence supports

The checked-in accepted snapshot supports describing BCDO as a structured specialist/generalist hybrid. It shows real wins over AdamW and several stress-task strengths while keeping a design that is visibly different from simple moment smoothing or coordinate-wise scaling.

The evidence also supports a more specific design statement: the accepted public line is the small-candidate, winner-take-all, typed-profile version of the method that survived branch pruning. That means block structure, matrix consensus, stable consensus, trusted-direction memory, and typed conv-safe scaling remain part of the public story, while heavier recoverability logic and other exploratory controls do not.

## What the current evidence does not support

This repository does **not** support claiming that BCDO is:

- a universal replacement for AdamW
- a universal replacement for RMSProp
- a universal replacement for SGD with momentum
- a strong public CNN optimizer yet
- a strong PINN optimizer
- a state-of-the-art optimizer claim

The accepted snapshot and the visible probe folders are important precisely because they keep those unsupported claims from creeping into the docs.

## Failure cases that should stay visible

The repo should keep saying plainly that:

- clean standard tasks often still favor simpler baselines
- the CNN gap remains open
- the PINN probe is a boundary condition, not a win surface
- runtime remains higher than the strongest simple baselines

## Unfair claims

It would be unfair to claim that the current runnable output path replaces the accepted public snapshot, that the reference CNN branch should be read as the public default, or that BCDO is already a general-purpose replacement for the best practical baselines.

## Safe public wording

The safest short sentence the repository currently supports is:

> BCDO is a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, or structurally misleading, with checked-in evidence of wins on stress-oriented and structure-sensitive slices, but not as a universal replacement for AdamW, RMSProp, SGD with momentum, CNN baselines, or PINN optimizers.
