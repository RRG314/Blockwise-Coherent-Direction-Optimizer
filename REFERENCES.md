# References

This repository compares the Blockwise Consensus Direction Optimizer (BCDO) against several optimizer families that already define the practical baseline for modern training. This file collects the external references used in the public README, method notes, and accepted reports so that the repository has one clear bibliography instead of repeating partial citation lists in multiple places.

All external URLs in this file were re-checked during the 2026-04-29 private GitHub preparation pass. When an optimizer does not have a single canonical archival paper, the repository uses the most widely cited primary source available and says so explicitly.

## SGD and momentum

For plain stochastic gradient descent, this repository uses Léon Bottou's survey-level treatment because it explains the modern machine-learning role of SGD directly rather than only the original stochastic-approximation theory. See:

- Léon Bottou, [Large-Scale Machine Learning with Stochastic Gradient Descent](https://leon.bottou.org/papers/bottou-2010), COMPSTAT 2010.

For momentum, the main point of comparison is that momentum still follows one recursively smoothed direction, even when it behaves very well in practice. The reference used here is:

- Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton, [On the Importance of Initialization and Momentum in Deep Learning](https://proceedings.mlr.press/v28/sutskever13.html), ICML 2013.

BCDO differs from SGD and momentum at the level of direction formation. SGD uses the instantaneous gradient. Momentum uses one running velocity. BCDO builds a small candidate set per block and then chooses among those candidates with a trust score.

## RMSProp

RMSProp does not have a standard archival conference paper in the same way Adam or AdamW do. The most common primary reference is Geoffrey Hinton's lecture note introducing the method:

- Geoffrey Hinton, [Neural Networks for Machine Learning, Lecture 6e: RMSProp](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf), 2012.

That matters here because RMSProp is one of the strongest practical baselines in the accepted BCDO runs. RMSProp rescales the gradient coordinate-wise using squared-gradient history. BCDO does not use coordinate-wise variance scaling to define its direction; it keeps direction choice and magnitude control separate.

## Adam and AdamW

The standard Adam reference in this repository is:

- Diederik P. Kingma and Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), 2014.

For AdamW, the repository uses the decoupled weight-decay paper rather than a framework documentation page:

- Ilya Loshchilov and Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019.

Adam and AdamW are important here because they show the standard pattern BCDO is trying not to collapse back into: start from one gradient direction, smooth it with a first-moment estimator, and scale it with a second-moment estimator. BCDO is not a modified AdamW controller. Its main decision is discrete blockwise direction selection, not adaptive moment transformation.

## Lion

The reference used for Lion is:

- Chen et al., [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675), 2023.

Lion is included as a compact low-state directional baseline. It still commits to one signed momentum direction per parameter, whereas BCDO treats signed or structured directions as candidates rather than as the only rule.

## Muon

The Muon-family reference used in this repository is:

- Xu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982), 2025.

Muon matters because it is a modern structure-aware optimizer that uses matrix orthogonalization aggressively. The comparison point for BCDO is deliberate: BCDO uses matrix structure to propose or score block directions, but it does not behave like a full orthogonalizing matrix optimizer.

## Shampoo

The Shampoo reference used here is:

- Vineet Gupta, Tomer Koren, and Yoram Singer, [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568), 2018.

Shampoo is relevant because it is a serious tensor-structured preconditioner rather than a lightweight first-order tweak. BCDO differs at the algorithmic level: it uses structure-aware candidate selection and bounded step scaling, not tensor-mode preconditioning.

## K-FAC

The K-FAC reference used in this repository is:

- James Martens and Roger Grosse, [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671), ICML 2015.

K-FAC is the clearest curvature-aware block baseline in the comparison set. BCDO does not estimate Fisher or curvature factors. If BCDO is useful, it is because simple blockwise direction selection can occasionally help without paying the state and compute cost of K-FAC-like methods.

## SAM and ASAM

For sharpness-aware methods, the repository uses:

- Pierre Foret et al., [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412), ICLR 2021.
- Jongheon Kwon et al., [ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks](https://arxiv.org/abs/2102.11600), 2021.

These methods matter because they change the effective training direction by optimizing against a perturbed neighborhood rather than the local point alone. BCDO uses no such min-max neighborhood objective in its accepted mainline. Its trust score is computed from local block signals and history.

## PCGrad and CAGrad

For explicit gradient-conflict methods, the repository uses:

- Tianhe Yu et al., [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782), NeurIPS 2020.
- Bo Liu et al., [Conflict-Averse Gradient Descent for Multi-task Learning](https://openreview.net/forum?id=_61Qh8tULj_), NeurIPS 2021.

These methods are relevant because BCDO also talks about conflict and oscillation, but the setting is different. PCGrad and CAGrad operate on multiple task gradients. BCDO does not require separate task losses; it scores candidate directions inside one block even in single-objective training.

## Lookahead

The reference used for Lookahead is:

- Michael R. Zhang et al., [Lookahead Optimizer: k steps forward, 1 step back](https://papers.nips.cc/paper/9155-lookahead-optimizer-k-steps-forward-1-step-back), NeurIPS 2019.

Lookahead is included as an example of a wrapper optimizer. BCDO is not a wrapper around another optimizer's inner loop. Its local rule is the direction-selection rule itself.

## Learned optimizers (VeLO)

For learned optimizers, the repository uses:

- Luke Metz et al., [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://arxiv.org/abs/2211.09760), 2022.

VeLO is not a baseline that BCDO tries to beat in the accepted suite. It appears in the literature-position notes because it is a clear example of a very different route: learn the update rule itself rather than hand-specifying it. BCDO is intentionally hand-specified and interpretable.

## Block coordinate methods

The repository uses one standard randomized block-coordinate reference:

- Peter Richtárik and Martin Takáč, [Iteration Complexity of Randomized Block-Coordinate Descent Methods for Minimizing a Composite Function](https://arxiv.org/abs/1107.2848), Mathematical Programming, 2014.

This reference matters because BCDO is blockwise, but it is not a block coordinate descent method in the classic sense. BCDO still updates all blocks each step; it does not choose only one coordinate block to update.

## Trust-region methods

The trust-region reference used in this repository is the standard monograph:

- Andrew R. Conn, Nicholas I. M. Gould, and Philippe L. Toint, [Trust Region Methods](https://books.google.com/books/about/Trust_Region_Methods.html?id=wfs-hsrd4WQC), SIAM, 2000.

This is included because BCDO uses trust-like quantities, but not in the classical trust-region sense. Classical trust-region methods decide how far to move along a proposed direction. BCDO uses trust scores to choose the direction first and then bounds the step second. The link above is the stable bibliographic preview page used for this repo because the SIAM-hosted book page did not provide a clean unauthenticated fetch path during the GitHub-preparation pass.

## CMA-ES

The repository uses the original covariance-matrix-adaptation paper as the main reference point:

- Nikolaus Hansen and Andreas Ostermeier, [Completely Derandomized Self-Adaptation in Evolution Strategies](https://docslib.org/doc/156104/completely-derandomized-self-adaptation-in-evolution-strategies), Evolutionary Computation, 2001.

CMA-ES appears only as a conceptual contrast. It samples directions from a search distribution. BCDO remains a first-order neural optimizer with a small deterministic candidate set. The link above is an accessible hosted copy of the original paper used because the MIT-hosted article route did not provide a clean anonymous fetch path during this documentation pass.

## Internal comparison baselines

Not every optimizer name appearing in the accepted reports is an external literature baseline. A few entries refer to repository-local companion baselines that were kept only to preserve the benchmark context:

- [Blockwise Consensus Direction Optimizer method notes](docs/METHOD.md): public method description for this repository.
- [Reference CNN branch notes](docs/CNN_REFERENCE.md): explanation of the retained CNN reference path used in the accepted comparison story.
- the internal coherent momentum baseline family: companion custom comparators used in broader internal benchmark tables. They are not part of the public BCDO surface, so this file treats them as context for the accepted reports rather than as literature-defined public baselines.

When the accepted BCDO reports mention that internal coherent momentum family, the purpose is only to preserve the benchmark context that produced the published CSVs and summary tables.
