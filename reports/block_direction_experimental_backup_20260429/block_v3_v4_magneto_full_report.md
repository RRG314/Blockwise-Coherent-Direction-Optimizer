# Full Report: BlockDirection V3, BlockDirection V4Fast, and Magneto-Hamiltonian Adam

## Scope

This report consolidates the current state of the three main custom optimizer branches:

- `BlockDirectionOptimizerV3`
- `BlockDirectionOptimizerV4Fast`
- `MagnetoHamiltonianAdam`

It is based only on the isolated branch runs and exported reports already completed in this workspace.

Source reports:

- [BlockDirectionOptimizerV3 report](</Users/stevenreid/Documents/New project/reports/block_direction_v3/final_report.md>)
- [BlockDirectionOptimizerV4Fast report](</Users/stevenreid/Documents/New project/reports/block_direction_v4_fast/final_report.md>)
- [BlockDirection upgrade decision](</Users/stevenreid/Documents/New project/reports/block_direction_upgrade_decision.md>)
- [Magneto-Hamiltonian Adam report](</Users/stevenreid/Documents/New project/reports/magneto_hamiltonian_adam/final_report.md>)

## Executive Summary

### Best current branch for the block family

`BlockDirectionOptimizerV4Fast` is the correct main block-direction branch going forward.

Why:

- it preserves the novel block-direction-selection principle
- it is substantially faster than V3
- it is much better than V3 on standard MLP-style tasks
- it keeps the main stress-task strengths that made the block branch interesting
- it improves low-rank and reversal-style tasks dramatically

### Best current custom optimizer overall

`MagnetoHamiltonianAdam` is still the strongest custom optimizer branch overall when the target regime is:

- oscillation
- direction reversal
- conflicting batches
- curved/noisy momentum-sensitive objectives

It is not the strongest default optimizer overall, but it is the strongest specialist custom branch at the moment.

### Publishability status

None of these is ready to claim broad default-optimizer superiority over `RMSProp`, `SGD+momentum`, or the strongest closure-based PINN baselines.

The most credible publishable direction right now is:

- `BlockDirectionOptimizerV4Fast` as a specialist structured direction-selection optimizer,
- if it can become clearly more competitive on CNNs and broader standard neural tasks without losing its stress-task edge.

## 1. Optimizer Definitions

### BlockDirectionOptimizerV3

V3 is a blockwise candidate-direction optimizer.

Core principle:

- split parameters into blocks
- generate multiple candidate directions per block
- score them by trust
- choose a direction at block level

V3 default structure:

- `winner_take_all` selection
- stable-consensus candidate
- row/column matrix-consensus candidate
- recoverability kept as a gate
- projection and orthogonal escape turned off by default but still available
- regime gating to suppress stress candidates on ordinary tasks

What makes it non-Adam:

- no Adam first-moment EMA defines the direction
- no Adam second-moment preconditioner defines the direction
- the core rule is blockwise direction choice

### BlockDirectionOptimizerV4Fast

V4Fast is the reduced hot-path successor to V3.

Core principle:

- same blockwise direction-selection idea
- smaller candidate set
- cheaper scoring path
- blockwise energy-normalized step magnitude

V4Fast default candidates:

- `gradient`
- `stable_consensus`
- `trusted_direction`
- `matrix_consensus`

V4Fast structural choices:

- `winner_take_all` default
- recoverability removed from the default per-candidate hot loop
- small tensors stay tensor-blocked
- larger matrices use row blocks
- convolutional kernels now reshape to `[out_channels, -1]` so filter-wise row/column consensus is meaningful

What makes it non-Adam:

- direction is still selected at block level
- second-moment-like energy is used only as a scalar block step normalizer, not as an Adam-style directional preconditioner

### MagnetoHamiltonianAdam

Magneto-Hamiltonian is a hybrid of the strengthened real Hamiltonian branch plus directional-coherence control.

Core principle:

- keep a Hamiltonian-style momentum/energy core
- modulate behavior using directional field signals

Key signals:

- gradient-momentum cosine
- force-momentum cosine
- gradient-history cosine
- update-history cosine
- rotation score
- coherence score

Core controls:

- projection back toward useful force direction during conflict
- damping under misalignment
- optional activation gating so magneto control ramps up mostly when conflict/rotation rises

What makes it different from Real HamiltonianAdam:

- stronger directional coherence layer
- better handling of reversal/conflict/oscillation regimes

## 2. What Changed In Each Branch

### V3 changes relative to earlier block versions

- simplified toward `winner_take_all`
- added stable consensus
- added row/column matrix consensus
- kept recoverability as a gate instead of a generator
- moved projection and orthogonal escape off the default path
- added regime gating so stress-specialist directions do not dominate ordinary tasks

### V4Fast changes relative to V3

- reduced default candidate set from a broad menu to four core candidates
- removed per-candidate recoverability from the hot path
- removed stress-specialist default machinery
- switched to `smart_v4` grouping
- added block-energy normalization
- later upgrade in this pass:
  - convolution/filter tensors now use filter-wise row/column consensus instead of collapsing to a single tensor block

### Magneto-Hamiltonian changes relative to Real Hamiltonian

- added directional coherence and rotation signals
- added projection toward force direction under conflict
- added activation gating to keep the branch closer to Real HamiltonianAdam on ordinary tasks and more magneto-like on conflict/rotation regimes

## 3. Core Math Summary

### V3 and V4 block math

For block `B_i`, define candidate directions `D_i`.

Stable consensus:

`d_i^(cons) = normalize(d_i^(grad) + a_i d_i^(trust) + b_i d_i^(smooth) + c_i d_i^(matrix))`

where each coefficient contributes only when the corresponding component aligns positively with descent.

Trust score:

`T_i(d) = w_d A_i(d) + w_m M_i(d) + w_q Q_i(d) + w_s S_i(d) + w_c C_i(d) - w_o O_i(d) - w_f F_i(d) - w_k cost(d)`

Terms:

- `A_i(d)`: descent alignment
- `M_i(d)`: memory coherence
- `Q_i(d)`: improvement-history memory
- `S_i(d)`: norm stability
- `C_i(d)`: consensus support
- `O_i(d)`: oscillation penalty
- `F_i(d)`: conflict penalty

Selection:

`d_i^* = argmax_d T_i(d)`

V4Fast step scaling:

`alpha_i = eta * lambda_i * ||g_i|| / (sqrt(E_i) + eps)^p`

where `E_i` is an EMA of block gradient energy and `lambda_i` is a bounded trust scale.

### Magneto-Hamiltonian math

Start with the stabilized real Hamiltonian core:

- physical momentum state
- adaptive Hamiltonian-style force
- damping/friction
- energy tracking

Then apply directional coherence controls using:

- `cos(g, m)`
- `cos(f, m)`
- `cos(g_t, g_{t-1})`
- `cos(u_t, u_{t-1})`
- rotation score

These signals modulate:

- effective damping
- projection strength
- alignment scaling

## 4. Mixed-Suite Results

### Block V3

Best reported row:

- `breast_cancer_mlp`: best val loss `0.239169`, best val acc `0.953125`

Comparison counts from the V3 report:

- vs `AdamW`: `4` meaningful wins
- vs `RMSProp`: `4`
- vs `SGD+momentum`: `3`
- vs `Muon hybrid`: `6`
- vs `RealHamiltonianAdam`: `5`
- vs `MagnetoHamiltonianAdam`: `4`

Runtime:

- mean runtime per step about `75.8 ms` in the V3 report

Interpretation:

- real specialist branch
- too slow
- too weak on standard tasks
- structure and stable consensus helped, but many extra rules did not

### Block V4Fast

Best reported row:

- `breast_cancer_mlp`: best val loss `0.061726`, best val acc `0.985677`

Comparison counts from the V4Fast report:

- vs `V3`: `8` meaningful wins
- vs `AdamW`: `7` meaningful wins, `5` tracked 2x events
- vs `RMSProp`: `6`
- vs `SGD+momentum`: `4`
- vs `MagnetoHamiltonianAdam`: `8`
- vs `RealHamiltonianAdam`: `8`

Runtime:

- mean runtime per step about `30.15 ms` in the main V4 report
- full benchmark average across tasks about `14.12 ms` per step

Interpretation:

- major improvement over V3
- still slower than standard optimizers
- still not a broad default winner
- real specialist, now actually usable

### Magneto-Hamiltonian Adam

Best reported row:

- `breast_cancer_mlp`: best val loss `0.0799`, best val acc `0.9805`

Comparison counts from the current Magneto report:

- vs `RealHamiltonianAdam`: `13` meaningful wins
- vs `MagnetoAdam`: `7`
- vs `AdamW`: `8`
- vs `RMSProp`: `4`
- vs `TopologicalAdam`: `8`

Best tasks where it beat Real Hamiltonian and AdamW while staying competitive with RMSProp:

- `direction_reversal_objective`
- `noisy_quadratic_objective`
- `oscillatory_valley`
- `quadratic_bowl_objective`
- `saddle_objective`

Interpretation:

- strongest custom optimizer branch overall for momentum/coherence stress regimes
- still not a general replacement for `RMSProp`

## 5. V3 vs V4 Direct Comparison

Shared mixed-suite block comparison:

| Task | V3 best loss | V3 best acc | V3 ms/step | V4 best loss | V4 best acc | V4 ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `breast_cancer_mlp` | 0.2392 | 0.9531 | 78.39 | 0.0617 | 0.9857 | 30.15 |
| `moons_mlp` | 0.3573 | 0.8601 | 77.03 | 0.1820 | 0.9289 | 26.99 |
| `wine_mlp` | 0.7339 | 0.9259 | 77.46 | 0.0580 | 0.9852 | 24.62 |
| `block_structure_classification` | 0.6033 | 0.7812 | 74.90 | 0.3618 | 0.8455 | 27.97 |
| `low_rank_matrix_objective` | 1.5310 | n/a | 13.80 | 0.0904 | n/a | 4.73 |
| `direction_reversal_objective` | 0.9657 | n/a | 10.94 | 0.0017 | n/a | 3.11 |
| `oscillatory_valley` | -0.071134 | n/a | 10.77 | -0.071087 | n/a | 3.24 |
| `plateau_escape_objective` | -1.050024 | n/a | 11.01 | -1.050024 | n/a | 3.21 |
| `saddle_objective` | -4.132232 | n/a | 10.86 | -4.132231 | n/a | 3.10 |

Conclusion:

- V4 is the correct branch to upgrade
- it did not lose the important stress-task capabilities
- it gained standard-task competitiveness and much better runtime
- it lost rule breadth, but mostly low-yield rule breadth

## 6. CNN / Standard Neural Probe

Focused neural probe for V4:

| Task | V4 best loss | V4 best acc | AdamW best acc | RMSProp best acc | SGD momentum best acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `breast_cancer_mlp` | 0.0623 | 0.9844 | 0.9857 | 0.9883 | 0.9831 |
| `moons_mlp` | 0.1820 | 0.9289 | 0.9074 | 0.9948 | 0.9810 |
| `wine_mlp` | 0.0580 | 0.9852 | 0.9778 | 0.9852 | 0.9852 |
| `digits_mlp` | 0.0774 | 0.9811 | 0.9779 | 0.9824 | 0.9792 |
| `digits_cnn` | 0.7047 | 0.7577 | 0.8863 | 0.9206 | 0.7577 |

Interpretation:

- V4 is now real and usable on dense small-neural tasks
- `digits_mlp` is competitive
- `digits_cnn` is still weak

Runtime in the neural probe:

- `V4Fast`: `26.42 ms/step`
- `V3`: `75.47 ms/step`
- `MagnetoHamiltonianAdam`: `7.08 ms/step`
- `AdamW`: `1.82 ms/step`
- `RMSProp`: `1.66 ms/step`
- `SGD+momentum`: `1.59 ms/step`

So V4 got much more usable, but still is not cheap enough to claim broad practical superiority.

## 7. PINN Probe

Focused PINN probe for V4:

| Task | V4 best loss | Strongest baseline |
| --- | ---: | --- |
| `pinn_harmonic_oscillator` | 0.3891 | `adamw_lbfgs_hybrid` at `0.0129` |
| `pinn_poisson_1d` | 0.0246 | `adamw_lbfgs_hybrid` at `1.3e-09` |
| `pinn_heat_equation` | 0.00889 | `lbfgs` at `6.0e-07` |

Interpretation:

- V4 beats V3 on PINNs
- V4 beats some simple first-order baselines
- V4 is not remotely competitive with `LBFGS` or `AdamW->LBFGS hybrid`

So PINNs are not the right publishable target domain for the block branch right now.

## 8. Which Rules Helped

### V3

From the V3 ablation:

- `stable_consensus` helped
- `block_structure` helped
- recoverability did not help broadly
- regime gating did not help broadly
- projection/orthogonal default behavior did not justify themselves

### V4Fast

From the V4 ablation:

- stable consensus helped
- matrix consensus helped most among the removable V4 rules
- trusted-direction memory helped slightly
- block structure helped slightly
- periodic recoverability hurt

### Magneto-Hamiltonian

From the current Magneto report:

- projection helped
- activation gating was neutral or harmful
- conflict damping hurt

## 9. What Is Missing

### For V4

Still missing for a strong claim:

- stronger CNN/filter behavior
- broader standard-task competitiveness against `RMSProp` and `SGD+momentum`
- an honest ARC benchmark

### For Magneto

Still missing:

- broader standard-task competitiveness
- a standard-safe preset that stays closer to the Real Hamiltonian core when conflict is low

### For both

Still missing:

- larger-scale benchmarks beyond current CPU-scale suites
- a benchmark domain where one of them clearly beats the strongest baseline family, not just isolated tasks

## 10. ARC Status

There is still **no credible local ARC benchmark** in the current optimizer harness.

So:

- there is no honest ARC performance claim for V3
- there is no honest ARC performance claim for V4
- there is no honest ARC performance claim for Magneto

Any ARC claim would currently be speculative.

## 11. Final Recommendation

### Block family

Use `BlockDirectionOptimizerV4Fast` as the main line.

Keep `BlockDirectionOptimizerV3` as:

- an experimental stress-specialist reference
- a sandbox for extra candidate rules that are too expensive for the fast branch

### Magneto family

Keep `MagnetoHamiltonianAdam` as the main Hamiltonian specialist branch.

Do not try to turn it into a broad default optimizer by stacking more controllers blindly.

### Most credible near-term publishable story

The most credible publishable direction right now is:

- a specialist optimizer story,
- not a universal optimizer story,
- centered on structured direction selection or momentum/coherence control under regimes where standard gradients conflict, rotate, or destabilize.

### Exact next steps

1. Upgrade `BlockDirectionOptimizerV4Fast` into `V4.1` with a CNN-specific filter/channel consensus improvement.
2. Re-test `digits_cnn`, `digits_mlp`, `breast_cancer_mlp`, `wine_mlp`, `moons_mlp`, `oscillatory_valley`, and `saddle_objective`.
3. Keep `MagnetoHamiltonianAdam` separate and add a stricter standard-safe preset.
4. Build a real ARC benchmark branch before making any ARC claims.

## Bottom Line

- `V4Fast` is better than `V3`.
- `V4Fast` did not merely look better; it actually improved standard-task behavior, runtime, and several structured/stress tasks.
- `MagnetoHamiltonianAdam` remains the strongest custom optimizer branch overall for its specialist regime.
- Neither branch is yet a broad replacement for `RMSProp` or `SGD+momentum`.
- The work is real and defensible, but the publishable claim should still be framed as **specialist advantage**, not universal superiority.
