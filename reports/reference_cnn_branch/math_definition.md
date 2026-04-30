# Reference CNN Branch Mathematical Definition

This branch is implemented internally as `BlockDirectionOptimizerV42`. External optimizer families named in the comparison notes are indexed in [../../REFERENCES.md](../../REFERENCES.md).

The reference CNN branch keeps the accepted BCDO candidate set

`D_i = {d_i^(grad), d_i^(cons), d_i^(trust), d_i^(matrix)}`

and changes only the conv-specific trust and step rules.

For convolutional tensors `G in R^{O x I x S}` with filters `G_o`, define a conv-structure support score

`R_o = 0.40 * align(G_o, channel_profile_o) + 0.35 * align(G_o, spatial_profile_o) + 0.25 * align(G_o, bank_profile)`

where each `align` term is the positive cosine agreement between the normalized filter direction and the corresponding broadcast profile.

The trust score becomes

`T_i(d) = T_i^(v4)(d) + b_c * R_i * C_i(d) + b_m * R_i * M_i(d)`

for the `stable_consensus` and `trusted_direction` candidates on conv tensors, and unchanged elsewhere.

Fallback on conv tensors becomes

`tau_i^(conv) = max(0, tau - rho * R_i)`

so coherent conv filters are less likely to fall back immediately to the raw gradient.

The conv-safe step scaling from V4.1 remains:

- lower conv update cap
- stronger energy-power term when conv support is weak
- bounded conv step multiplier
