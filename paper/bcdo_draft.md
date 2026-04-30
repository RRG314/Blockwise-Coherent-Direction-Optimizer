# Blockwise Consensus Direction Optimizer: Blockwise Direction Selection Under Unstable Local Geometry

## Abstract

This draft documents the current public claim supported by the repository: Blockwise Consensus Direction Optimizer is a blockwise direction-selection optimizer for tasks where the raw gradient direction is unstable, conflicting, structurally misleading, or too myopic to trust block-for-block. The method forms a small candidate set per parameter block, scores those candidates with structural trust signals, selects one winner, and then applies a bounded blockwise step. The checked-in accepted snapshot supports a narrow claim. BCDO shows meaningful wins over AdamW and several structure-sensitive or stress-oriented strengths, but it remains slower than simpler baselines and does not support broad claims on CNNs or PINNs.

## Introduction

Most first-order optimizers accept the current gradient direction and focus on smoothing, scaling, or preconditioning that direction. BCDO treats the problem differently. It asks whether the block should follow the raw gradient at all, or whether a consensus, memory, or matrix-structured candidate is more trustworthy for that block at that step.

This shift is the main reason the repository keeps BCDO separate from AdamW-style adaptive methods. The public story is not that BCDO has a slightly different adaptive denominator. The story is that it chooses between competing block directions before it scales the step.

## Related Work

The closest baseline families are SGD and momentum, RMSProp, Adam and AdamW, and structure-aware optimizers such as Muon, Shampoo, and K-FAC. BCDO shares the first-order training-loop footprint of the simpler baselines, but it uses block structure for direction choice rather than for tensor preconditioning. SAM and ASAM matter as stability comparisons, while PCGrad and CAGrad matter as conflict-aware references, but neither family uses the same blockwise candidate-selection rule.

## Method

Let `g_i` be the gradient restricted to block `i`. BCDO forms a small candidate set `D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`, scores each candidate with a trust function that combines descent alignment, memory coherence, quality memory, norm stability, consensus support, oscillation penalty, conflict penalty, and a practical cost term, and selects the winning direction with

`d_i* = argmax_d T_i(d)`.

After direction choice, the optimizer applies a bounded block-energy step

`alpha_i = lr * trust_i * ||g_i|| / (sqrt(E_i) + eps)^p`.

This keeps direction choice and magnitude control separate. That separation is the main algorithmic difference between BCDO and standard adaptive baselines.

## Experimental Setup

This draft uses only local checked-in artifacts:

- accepted public snapshot: `reports/accepted_bcdo/benchmark_results.csv`
- current runnable output path: `reports/bcdo_mainline/benchmark_results.csv`
- reference CNN branch: `reports/reference_cnn_branch/benchmark_results.csv`
- CNN probe: `reports/bcdo_cnn_probe/benchmark_results.csv`
- PINN probe: `reports/bcdo_pinn_probe/benchmark_results.csv`

The accepted public snapshot and the current runnable output path are kept separate on purpose. The accepted snapshot is the fixed first-release public evidence line. The runnable directory is the writable location used by local scripts.

## Results

The best accepted BCDO row is `breast_cancer_mlp` with mean best validation loss `0.061726` and mean best validation accuracy `0.985677`. The accepted snapshot shows `7` meaningful wins over AdamW, `6` over RMSProp, `4` over SGD with momentum, and `10` over the internal Coherent Momentum comparator.

These results support a narrow claim: blockwise candidate selection can be useful on structure-sensitive and stress-oriented slices. They do not support a claim that BCDO replaces the strongest simple baselines broadly.

## Ablations

The accepted ablation report is important because it shows that the public line is not just a random accumulation of branch logic. Block structure, matrix consensus, typed conv and dense profiles, cheap conv structure support, stable consensus, and trusted-direction memory remain the parts that survive the public line. Recoverability did not survive the accepted default path. The public method should therefore be described as the small candidate-set selector that remained after branch pruning, not as the union of every explored idea.

## Runtime and Memory Costs

The accepted public line currently averages `24.0797 ms` per step against `1.3407 ms` for AdamW in the checked-in snapshot. That runtime cost narrows the set of workloads where the blockwise logic is worth using. The method only makes sense when blockwise directional reliability is a real bottleneck.

## Failure Cases

The CNN and PINN probe folders remain part of the paper package because they show where BCDO is still weak. The reference CNN branch contains useful conv-aware ideas, but the public accepted line is not a CNN-leading optimizer. The PINN probe likewise shows closure-friendly baselines ahead.

## Limitations

The current claim is deliberately narrow. BCDO is not a universal optimizer, not a dominant CNN optimizer, and not a PINN optimizer based on the checked-in evidence. The accepted public line is also slower than the strongest simple baselines.

## Reproducibility Statement

The repository includes focused tests, small runnable examples, a `build_paper_artifacts.py` script that regenerates the paper tables and figures from local CSVs, and a `run_paper_smoke.py` script that checks the paper package without rerunning the full expensive suite.

## Conclusion

The current repository supports a paper-scale draft only if the claim stays narrow. BCDO is best described as a blockwise direction-selection optimizer for workloads where the raw gradient direction is not trustworthy block-for-block. That is enough to make the method interesting. It is not enough to justify a broad default-optimizer story.

## Figures and Tables

- Accepted snapshot summary: `paper/tables/accepted_bcdo_summary.md`
- Accepted vs runnable split: `paper/tables/accepted_vs_runnable_summary.md`
- Reference CNN branch summary: `paper/tables/reference_cnn_summary.md`
- CNN probe summary: `paper/tables/cnn_probe_summary.md`
- PINN probe summary: `paper/tables/pinn_probe_summary.md`
- Ablation summary: `paper/tables/accepted_ablation_summary.md`
- Figures: `paper/figures/`

## References

- Bottou, Léon. “Large-Scale Machine Learning with Stochastic Gradient Descent.” 2010. <https://leon.bottou.org/papers/bottou-2010>
- Sutskever, Ilya, James Martens, George Dahl, and Geoffrey Hinton. “On the Importance of Initialization and Momentum in Deep Learning.” ICML 2013. <https://proceedings.mlr.press/v28/sutskever13.html>
- Hinton, Geoffrey. “Neural Networks for Machine Learning, Lecture 6e.” 2012. <https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf>
- Kingma, Diederik P., and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” 2015. <https://arxiv.org/abs/1412.6980>
- Loshchilov, Ilya, and Frank Hutter. “Decoupled Weight Decay Regularization.” 2019. <https://arxiv.org/abs/1711.05101>
- Gupta, Vineet, Tomer Koren, and Yoram Singer. “Shampoo: Preconditioned Stochastic Tensor Optimization.” 2018. <https://arxiv.org/abs/1802.09568>
- Martens, James, and Roger Grosse. “Optimizing Neural Networks with Kronecker-factored Approximate Curvature.” 2015. <https://arxiv.org/abs/1503.05671>
- Foret, Pierre, et al. “Sharpness-Aware Minimization for Efficiently Improving Generalization.” 2021. <https://arxiv.org/abs/2010.01412>
- Yu, Tianhe, et al. “Gradient Surgery for Multi-Task Learning.” 2020. <https://arxiv.org/abs/2001.06782>
