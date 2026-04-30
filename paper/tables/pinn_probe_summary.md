# Pinn Probe Summary

Focused PINN probe used to keep closure-friendly failure cases visible.

Source CSVs:
- `reports/bcdo_pinn_probe/benchmark_results.csv`

| task | optimizer | mean_best_val_loss | mean_runtime_per_step_ms | source_csv |
| --- | --- | --- | --- | --- |
| pinn_harmonic_oscillator | adamw | 0.4347 | 1.6996 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_harmonic_oscillator | adamw_lbfgs_hybrid | 0.0129 | 11.9604 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_harmonic_oscillator | blockwise_consensus_direction_optimizer | 0.3891 | 24.5157 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_harmonic_oscillator | lbfgs | 0.0193 | 14.1256 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_harmonic_oscillator | rmsprop | 0.3925 | 1.5508 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_harmonic_oscillator | sgd_momentum | inf | 1.4908 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_heat_equation | adamw | 0.0003 | 2.8947 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_heat_equation | adamw_lbfgs_hybrid | 0.0000 | 21.7278 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_heat_equation | blockwise_consensus_direction_optimizer | 0.0089 | 34.8671 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_heat_equation | lbfgs | 0.0000 | 26.6015 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_heat_equation | rmsprop | 0.0131 | 2.7953 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_heat_equation | sgd_momentum | inf | 2.4920 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_poisson_1d | adamw | 0.0000 | 2.2360 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_poisson_1d | adamw_lbfgs_hybrid | 0.0000 | 17.5556 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_poisson_1d | blockwise_consensus_direction_optimizer | 0.0246 | 29.4689 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_poisson_1d | lbfgs | 0.0000 | 20.6231 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_poisson_1d | rmsprop | 0.0539 | 2.0753 | reports/bcdo_pinn_probe/benchmark_results.csv |
| pinn_poisson_1d | sgd_momentum | inf | 1.9098 | reports/bcdo_pinn_probe/benchmark_results.csv |