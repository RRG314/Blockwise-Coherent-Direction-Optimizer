# Reference Cnn Summary

Reference CNN branch summary kept visible as benchmark context rather than as the public identity of the repo.

Source CSVs:
- `reports/reference_cnn_branch/benchmark_results.csv`

| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | source_csv |
| --- | --- | --- | --- | --- | --- |
| breast_cancer_mlp | adamw | 0.0615 | 0.9857 | 1.0949 | reports/reference_cnn_branch/benchmark_results.csv |
| breast_cancer_mlp | bcdo_cnn_reference | 0.0583 | 0.9883 | 33.3712 | reports/reference_cnn_branch/benchmark_results.csv |
| breast_cancer_mlp | blockwise_consensus_direction_optimizer | 0.0597 | 0.9870 | 27.7092 | reports/reference_cnn_branch/benchmark_results.csv |
| breast_cancer_mlp | rmsprop | 0.0567 | 0.9883 | 0.9567 | reports/reference_cnn_branch/benchmark_results.csv |
| breast_cancer_mlp | sgd_momentum | 0.0617 | 0.9831 | 0.9035 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_cnn | adamw | 0.4022 | 0.8658 | 4.9882 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_cnn | bcdo_cnn_reference | 0.3360 | 0.8797 | 42.4673 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_cnn | blockwise_consensus_direction_optimizer | 0.4409 | 0.8377 | 35.0186 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_cnn | rmsprop | 0.2735 | 0.9166 | 4.8629 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_cnn | sgd_momentum | 0.4848 | 0.8319 | 4.7302 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_mlp | adamw | 0.0754 | 0.9779 | 1.3500 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_mlp | bcdo_cnn_reference | 0.0962 | 0.9812 | 34.7545 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_mlp | blockwise_consensus_direction_optimizer | 0.0928 | 0.9812 | 27.1819 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_mlp | rmsprop | 0.0834 | 0.9826 | 1.2147 | reports/reference_cnn_branch/benchmark_results.csv |
| digits_mlp | sgd_momentum | 0.0711 | 0.9805 | 1.1099 | reports/reference_cnn_branch/benchmark_results.csv |
| oscillatory_valley | adamw | -0.0711 | nan | 0.3753 | reports/reference_cnn_branch/benchmark_results.csv |
| oscillatory_valley | bcdo_cnn_reference | -0.0711 | nan | 3.7275 | reports/reference_cnn_branch/benchmark_results.csv |
| oscillatory_valley | blockwise_consensus_direction_optimizer | -0.0711 | nan | 3.0339 | reports/reference_cnn_branch/benchmark_results.csv |
| oscillatory_valley | rmsprop | -0.0711 | nan | 0.2516 | reports/reference_cnn_branch/benchmark_results.csv |
| oscillatory_valley | sgd_momentum | -0.0711 | nan | 0.2327 | reports/reference_cnn_branch/benchmark_results.csv |
| plateau_escape_objective | adamw | -0.9799 | nan | 0.3231 | reports/reference_cnn_branch/benchmark_results.csv |
| plateau_escape_objective | bcdo_cnn_reference | -1.0500 | nan | 3.7325 | reports/reference_cnn_branch/benchmark_results.csv |
| plateau_escape_objective | blockwise_consensus_direction_optimizer | -1.0500 | nan | 3.1093 | reports/reference_cnn_branch/benchmark_results.csv |
| plateau_escape_objective | rmsprop | -1.0500 | nan | 0.2850 | reports/reference_cnn_branch/benchmark_results.csv |
| plateau_escape_objective | sgd_momentum | -1.0500 | nan | 0.2746 | reports/reference_cnn_branch/benchmark_results.csv |
| saddle_objective | adamw | -4.1217 | nan | 0.2575 | reports/reference_cnn_branch/benchmark_results.csv |
| saddle_objective | bcdo_cnn_reference | -4.1322 | nan | 3.6951 | reports/reference_cnn_branch/benchmark_results.csv |
| saddle_objective | blockwise_consensus_direction_optimizer | -4.1322 | nan | 3.0075 | reports/reference_cnn_branch/benchmark_results.csv |
| saddle_objective | rmsprop | -4.1322 | nan | 0.2249 | reports/reference_cnn_branch/benchmark_results.csv |
| saddle_objective | sgd_momentum | -4.1322 | nan | 0.2138 | reports/reference_cnn_branch/benchmark_results.csv |