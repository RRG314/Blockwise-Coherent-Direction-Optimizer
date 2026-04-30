# Failure Cases

References for the external optimizers mentioned here are collected in [../REFERENCES.md](../REFERENCES.md).

BCDO is easier to understand if the repo says plainly where it is not the right choice.

## Clean standard tasks

On ordinary clean supervised problems, especially the smaller MLP tasks in the accepted suite, the main BCDO question can simply be the wrong question. If the raw direction is already usable, then candidate selection adds overhead before it adds value.

In those cases the practical baselines often remain stronger:

- `RMSProp`
- `SGD+momentum`
- sometimes `AdamW`

This is why the accepted public line is framed as a structured specialist/generalist hybrid rather than as a new default optimizer.

## CNNs

The repo keeps the reference CNN branch and the focused CNN probe visible because the mainline CNN story is still incomplete. Conv-safe scaling and typed conv handling helped enough to matter, but the accepted mainline still trails the strongest practical CNN baselines in the checked-in reports.

The stronger practical choices are still often:

- `RMSProp`
- `SGD+momentum`
- `AdamW`

The reference CNN branch is useful because it shows where conv-aware trust may help, but it does not close the public gap by itself.

## PINNs and closure-heavy scientific tasks

The PINN probe is included to show the boundary of the method, not to market it into a domain where the checked-in results are weak. The repository does not currently support a strong public claim for BCDO on PINN-style workloads.

Stronger choices there are likely to be:

- `LBFGS`
- `AdamW -> LBFGS` hybrid schedules

## Stable smooth problems

If a problem is already smooth and directionally well behaved, BCDO may be solving a problem that is not actually present. In that setting the mainline direction-selection logic can be more machinery than the task needs.

For those workloads, simpler methods are often the better tool:

- `SGD+momentum` if you want speed and a clean baseline
- `RMSProp` if you want a strong small-task adaptive baseline
- `AdamW` if you want a familiar general-purpose adaptive baseline

## Runtime

The accepted public line is still slower than the strongest simple baselines. That does not make the optimizer invalid, but it does narrow the set of workloads where the blockwise logic pays for itself. If throughput is the main priority and the task does not show clear directional instability or block-structure sensitivity, BCDO is usually the wrong first choice.

## What this file is for

This file is here to keep the public claim honest. BCDO does not need to win every workload to be a meaningful optimizer idea, but it does need its failure boundaries to stay visible in the repo.
