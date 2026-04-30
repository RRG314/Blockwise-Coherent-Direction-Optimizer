# BCDO CNN Probe Literature Scan

Full citations for every optimizer family mentioned here are collected in [../../REFERENCES.md](../../REFERENCES.md). This file keeps the comparison matrix concise and uses the reference index as the bibliography.

This probe keeps the same literature pressure as the accepted BCDO line: it is only interesting if blockwise direction **selection** does useful work that standard gradient **transforms** are not already doing.

| family | representative_method | direction_or_transform | scope | how_bcdo_differs | required_baseline |
| --- | --- | --- | --- | --- | --- |
| Momentum / SGD | SGD with momentum | Transforms gradient into velocity; does not choose among multiple candidate directions. | Global / per-parameter | BCDO explicitly selects blockwise directions from a candidate pool instead of using one momentum recursion. | sgd_momentum |
| RMSProp | RMSProp | Per-parameter scaling of the gradient; no candidate selection. | Per-parameter | BCDO keeps step magnitude separate and focuses on blockwise direction trust rather than coordinate-wise variance scaling. | rmsprop |
| Adam / AdamW | AdamW | Transforms the gradient through EMA momentum and variance normalization. | Per-parameter | BCDO is not an Adam-family moment method; it stores block trust and direction memories instead of Adam moments. | adamw |
| Lion | Lion | Transforms the gradient into a signed momentum direction. | Per-parameter | BCDO chooses blockwise directions from several structured candidates instead of committing to a single signed-momentum rule. | lion |
| Muon / matrix orthogonalization | Muon | Transforms matrix gradients into orthogonalized matrix directions. | Matrix-wise / blockwise | BCDO may include a Muon-like candidate, but only as one option inside a broader blockwise trust-and-selection rule. | muon_hybrid |
| Shampoo | Shampoo | Transforms gradients with block/tensor preconditioners. | Tensor / block-wise | BCDO is not a tensor preconditioner; it is a candidate-direction selector with lightweight block memory. | muon_hybrid |
| K-FAC / natural gradient | K-FAC | Transforms gradients with an approximate inverse curvature matrix. | Layer / block-wise | BCDO does not estimate curvature; it evaluates candidate directions by trust signals such as coherence and recoverability. | adamw |
| SAM / ASAM | SAM | Transforms the objective / gradient using a perturb-and-recompute step. | Global / layer-wise perturbation | BCDO uses cheap blockwise perturbation only to gate candidate trust, not to redefine the full objective. | adamw |
| Gradient surgery | PCGrad | Selects/projectively edits a gradient combination across task losses. | Task-global / shared-parameter | BCDO works with single-task blocks too and compares a richer candidate pool than pairwise task projections. | magneto_hamiltonian_adam |
| Conflict-aware MTL | CAGrad | Chooses a joint gradient direction under conflict constraints. | Task-global | BCDO operates blockwise and does not require multiple explicit task gradients to define its trust rule. | magneto_hamiltonian_adam |
| Lookahead | Lookahead | Wraps another optimizer rather than defining a new local direction rule. | Global | BCDO makes block-local direction choices directly; it is not a two-timescale wrapper. | adamw |
| Learned optimizers | VeLO | Learns update transformations from data instead of hand-specifying the rule. | Global / meta-learned | BCDO is hand-specified and interpretable, with explicit block trust components instead of a learned policy. | adamw |
| Block coordinate / direction methods | Block coordinate descent | Selects coordinates/blocks, often with simple local descent rules. | Coordinate / block-wise | BCDO updates all blocks but selects different candidate directions inside each block based on trust. | sgd_momentum |
| Trust-region methods | Trust-region gradient methods | Uses trust to scale or reject steps. | Global / block-wise depending on method | BCDO uses trust to rank discrete candidate directions inside each block, not just to scale one search direction. | adamw |
| Evolutionary / black-box search | CMA-ES | Samples candidate directions rather than transforming the gradient. | Global population-based | BCDO is still first-order and cheap enough for neural training; it scores a small deterministic candidate set per block. | sgd_momentum |

## Probe-specific interpretation

- Muon, Shampoo, and K-FAC are still the main structure-aware baseline pressure.
- SAM remains the most relevant perturbation-trust baseline, but the accepted BCDO mainline does not perturb the objective itself.
- The novelty opening is still the same narrow one: blockwise candidate-direction choice with lightweight structure and memory, not another adaptive-gradient transform.
