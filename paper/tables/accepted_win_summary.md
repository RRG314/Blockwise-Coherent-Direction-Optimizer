# Accepted Win Summary

Meaningful win counts for the accepted public snapshot, computed from the checked-in benchmark CSV.

Source CSVs:
- `reports/accepted_bcdo/benchmark_results.csv`

| benchmark_group | optimizer | baseline | meaningful_wins | two_x_wins | source_csv |
| --- | --- | --- | --- | --- | --- |
| accepted_public_snapshot | blockwise_consensus_direction_optimizer | bcdo_mainline_legacy | 4 | 0 | reports/accepted_bcdo/benchmark_results.csv |
| accepted_public_snapshot | blockwise_consensus_direction_optimizer | bcdo_cnn_reference | 4 | 0 | reports/accepted_bcdo/benchmark_results.csv |
| accepted_public_snapshot | blockwise_consensus_direction_optimizer | adamw | 7 | 5 | reports/accepted_bcdo/benchmark_results.csv |
| accepted_public_snapshot | blockwise_consensus_direction_optimizer | rmsprop | 6 | 0 | reports/accepted_bcdo/benchmark_results.csv |
| accepted_public_snapshot | blockwise_consensus_direction_optimizer | sgd_momentum | 4 | 1 | reports/accepted_bcdo/benchmark_results.csv |
| accepted_public_snapshot | blockwise_consensus_direction_optimizer | magneto_hamiltonian_adam | 10 | 1 | reports/accepted_bcdo/benchmark_results.csv |