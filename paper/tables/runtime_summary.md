# Runtime Summary

Runtime summary aggregated from the accepted public snapshot.

Source CSVs:
- `reports/accepted_bcdo/benchmark_results.csv`

| optimizer | mean_runtime_per_step_ms | mean_optimizer_state_mb | source_csv |
| --- | --- | --- | --- |
| rmsprop | 1.2613 | 0.0086 | reports/accepted_bcdo/benchmark_results.csv |
| sgd_momentum | 1.2992 | 0.0085 | reports/accepted_bcdo/benchmark_results.csv |
| adamw | 1.3407 | 0.0171 | reports/accepted_bcdo/benchmark_results.csv |
| magneto_hamiltonian_adam | 4.8806 | 0.0512 | reports/accepted_bcdo/benchmark_results.csv |
| bcdo_cnn_reference | 22.8265 | 0.0360 | reports/accepted_bcdo/benchmark_results.csv |
| bcdo_mainline_legacy | 23.8415 | 0.0360 | reports/accepted_bcdo/benchmark_results.csv |
| blockwise_consensus_direction_optimizer | 24.0797 | 0.0360 | reports/accepted_bcdo/benchmark_results.csv |