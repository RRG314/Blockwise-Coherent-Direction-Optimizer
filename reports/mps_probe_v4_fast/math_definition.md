# BlockDirectionOptimizerV4Fast Mathematical Definition

Split parameters into blocks `B_i` and compute block gradients `g_i`.

## Candidate set

V4Fast keeps a small candidate set per block:

`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`

where:

- `d_i^(grad)` is the normalized negative gradient direction
- `d_i^(trust)` is the normalized trusted-direction memory
- `d_i^(matrix)` is a structured row/column matrix consensus direction for 2D tensors
- `d_i^(cons)` is a stable consensus direction built from the candidates that currently agree

For the consensus candidate:

`d_i^(cons) = normalize(d_i^(grad) + a_i d_i^(trust) + b_i d_i^(smooth) + c_i d_i^(matrix))`

with coefficients only contributing when they align positively with the current descent direction.

## Trust score

Each candidate receives

`T_i(d) = w_d A_i(d) + w_m M_i(d) + w_q Q_i(d) + w_s S_i(d) + w_c C_i(d) - w_o O_i(d) - w_f F_i(d) - w_k cost(d)`

where:

- `A_i(d)` is descent alignment with the current negative gradient direction
- `M_i(d)` is coherence with trusted and smoothed direction memory
- `Q_i(d)` is candidate improvement-history memory
- `S_i(d)` is norm stability relative to block gradient EMA
- `C_i(d)` is consensus support
- `O_i(d)` is oscillation against previous gradient and update directions
- `F_i(d)` is conflict against trusted direction and previous update memory

## Selection

V4Fast defaults to

`d_i^* = argmax_{d in D_i} T_i(d)`

with winner-take-all selection.

## Step magnitude

The chosen direction is scaled by blockwise energy normalization:

`alpha_i = eta * lambda_i * ||g_i|| / (sqrt(E_i) + eps)^p`

where `E_i` is an EMA of mean squared gradient energy in block `i`, `p` is `energy_power`, and `lambda_i` is the bounded trust scale. A parameter-relative cap prevents destructive overshoot.

## Why V4Fast is still non-Adam

- No Adam first-moment EMA chooses the direction.
- No Adam per-coordinate second-moment preconditioner chooses the direction.
- The gradient-energy EMA only scales block step size; the novelty remains blockwise direction selection.
