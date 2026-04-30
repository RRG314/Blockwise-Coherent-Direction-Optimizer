# BlockDirectionOptimizerV4Fast Literature Scan

V4Fast keeps the same literature pressure as the V2 and V3 block-direction branch: it is only interesting if blockwise direction **selection** does useful work that standard gradient **transforms** are not already doing.

| family | representative_method | direction_or_transform | scope | difference_from_block_direction_v2 | required_baseline |
| --- | --- | --- | --- | --- | --- |
| Momentum / SGD | SGD with momentum | Transforms gradient into velocity; does not choose among multiple candidate directions. | Global / per-parameter | V2 explicitly selects blockwise directions from a candidate pool instead of using one momentum recursion. | sgd_momentum |
| RMSProp | RMSProp | Per-parameter scaling of the gradient; no candidate selection. | Per-parameter | V2 keeps step magnitude separate and focuses on blockwise direction trust rather than coordinate-wise variance scaling. | rmsprop |
| Adam / AdamW | AdamW | Transforms the gradient through EMA momentum and variance normalization. | Per-parameter | V2 is not an Adam-family moment method; it stores block trust and direction memories instead of Adam moments. | adamw |
| Lion | Lion | Transforms the gradient into a signed momentum direction. | Per-parameter | V2 chooses blockwise directions from several structured candidates instead of committing to a single signed-momentum rule. | lion |
| Muon / matrix orthogonalization | Muon | Transforms matrix gradients into orthogonalized matrix directions. | Matrix-wise / blockwise | V2 may include a Muon-like candidate, but only as one option inside a broader blockwise trust-and-selection rule. | muon_hybrid |
| Shampoo | Shampoo | Transforms gradients with block/tensor preconditioners. | Tensor / block-wise | V2 is not a tensor preconditioner; it is a candidate-direction selector with lightweight block memory. | muon_hybrid |
| K-FAC / natural gradient | K-FAC | Transforms gradients with an approximate inverse curvature matrix. | Layer / block-wise | V2 does not estimate curvature; it evaluates candidate directions by trust signals such as coherence and recoverability. | adamw |
| SAM / ASAM | SAM | Transforms the objective / gradient using a perturb-and-recompute step. | Global / layer-wise perturbation | V2 uses cheap blockwise perturbation only to gate candidate trust, not to redefine the full objective. | adamw |
| Gradient surgery | PCGrad | Selects/projectively edits a gradient combination across task losses. | Task-global / shared-parameter | V2 works with single-task blocks too and compares a richer candidate pool than pairwise task projections. | magneto_hamiltonian_adam |
| Conflict-aware MTL | CAGrad | Chooses a joint gradient direction under conflict constraints. | Task-global | V2 operates blockwise and does not require multiple explicit task gradients to define its trust rule. | magneto_hamiltonian_adam |
| Lookahead | Lookahead | Wraps another optimizer rather than defining a new local direction rule. | Global | V2 makes block-local direction choices directly; it is not a two-timescale wrapper. | adamw |
| Learned optimizers | VeLO | Learns update transformations from data instead of hand-specifying the rule. | Global / meta-learned | V2 is hand-specified and interpretable, with explicit block trust components instead of a learned policy. | adamw |
| Block coordinate / direction methods | Block coordinate descent | Selects coordinates/blocks, often with simple local descent rules. | Coordinate / block-wise | V2 updates all blocks but selects different candidate directions inside each block based on trust. | sgd_momentum |
| Trust-region methods | Trust-region gradient methods | Uses trust to scale or reject steps. | Global / block-wise depending on method | V2 uses trust to rank discrete candidate directions inside each block, not just to scale one search direction. | adamw |
| Evolutionary / black-box search | CMA-ES | Samples candidate directions rather than transforming the gradient. | Global population-based | V2 is still first-order and cheap enough for neural training; it scores a small deterministic candidate set per block. | sgd_momentum |

## V4Fast-specific interpretation

- Muon, Shampoo, and K-FAC are still the main structure-aware baseline pressure.
- SAM remains the most relevant perturbation-trust baseline, but V4Fast does not perturb the objective itself.
- The novelty opening is still the same narrow one: blockwise candidate-direction choice with lightweight structure and memory, not another adaptive-gradient transform.

- [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.html)
- [Neural Networks for Machine Learning, Lecture 6e](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)
- [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671)
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
- [Conflict-Averse Gradient Descent for Multi-task learning](https://openreview.net/forum?id=_61Qh8tULj_)
- [Lookahead Optimizer: k steps forward, 1 step back](https://papers.nips.cc/paper_files/paper/2019/file/90fd4f88f588ae64038134f1eeaa023f-Paper.pdf)
- [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://arxiv.org/abs/2211.09760)
- [Efficiency of Coordinate Descent Methods on Huge-Scale Optimization Problems](https://link.springer.com/article/10.1007/s10107-012-0597-5)
- [Trust Region Methods](https://epubs.siam.org/doi/book/10.1137/1.9780898719857)
- [Completely Derandomized Self-Adaptation in Evolution Strategies](https://link.springer.com/article/10.1023/A:1009669705971)
