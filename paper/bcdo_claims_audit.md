# BCDO Claims Audit

This file is the paper-facing claim check for the public Blockwise Consensus Direction Optimizer line.

## Supported claims

The checked-in accepted snapshot supports presenting BCDO as a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, or structurally misleading at the block level. The accepted first-release line still shows meaningful wins over AdamW and real stress-task strength while keeping the design materially different from moment-only adaptive baselines.

In the accepted public snapshot BCDO shows `7` meaningful wins over AdamW, `6` over RMSProp, and `4` over SGD with momentum. Those counts are enough to justify a narrow structured-specialist claim.

## Claims the repository does not support

The repository does not support a universal optimizer claim, a broad CNN claim, or a PINN claim. It also does not support claiming that every branch experiment should be promoted into the public line. The accepted snapshot stays separate for that reason.

## Failure cases that should remain visible

The CNN probe still shows the practical baselines ahead on the digits CNN slice, and the PINN probe still favors closure-based alternatives such as LBFGS or AdamW-to-LBFGS hybrids. Those are boundary conditions, not footnotes.

## Unfair claims

It would be unfair to claim that BCDO is a strong CNN optimizer, a PINN optimizer, or a general replacement for RMSProp, SGD with momentum, or AdamW. It would also be unfair to blur the accepted public snapshot together with later runnable or exploratory output folders.

## Safe public wording

BCDO is a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, or structurally misleading. The accepted snapshot shows real wins on stress-oriented and structure-sensitive slices, but it does not replace the strongest simple baselines as a general default optimizer.

## Runtime context

The accepted public line currently averages `24.0797 ms` per step against `1.3407 ms` for AdamW in the checked-in snapshot. That runtime gap is part of the honest public story.

## Probe reminders

The CNN and PINN probes remain visible because they show where the method is still weak. Hiding those reports would make the repository less credible, not more.

## Sources

- `reports/accepted_bcdo/benchmark_results.csv`
- `reports/accepted_bcdo/win_flags.csv`
- `reports/reference_cnn_branch/benchmark_results.csv`
- `reports/bcdo_cnn_probe/benchmark_results.csv`
- `reports/bcdo_pinn_probe/benchmark_results.csv`