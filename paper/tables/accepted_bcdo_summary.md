# Accepted Bcdo Summary

Accepted first-release BCDO rows aggregated from the checked-in public snapshot.

Source CSVs:
- `reports/accepted_bcdo/benchmark_results.csv`

| task | optimizer | task_family | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_optimizer_state_mb | source_csv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer_mlp | adamw | neural | 0.0615 | 0.9857 | 1.3174 | 0.0297 | reports/accepted_bcdo/benchmark_results.csv |
| breast_cancer_mlp | bcdo_cnn_reference | neural | 0.0622 | 0.9844 | 39.8155 | 0.0635 | reports/accepted_bcdo/benchmark_results.csv |
| breast_cancer_mlp | bcdo_mainline_legacy | neural | 0.0617 | 0.9857 | 41.5548 | 0.0635 | reports/accepted_bcdo/benchmark_results.csv |
| breast_cancer_mlp | blockwise_consensus_direction_optimizer | neural | 0.0617 | 0.9857 | 39.5980 | 0.0635 | reports/accepted_bcdo/benchmark_results.csv |
| breast_cancer_mlp | rmsprop | neural | 0.0567 | 0.9883 | 1.1667 | 0.0149 | reports/accepted_bcdo/benchmark_results.csv |
| breast_cancer_mlp | sgd_momentum | neural | 0.0634 | 0.9870 | 1.0993 | 0.0148 | reports/accepted_bcdo/benchmark_results.csv |
| digits_cnn | adamw | neural | 0.3787 | 0.8856 | 6.5117 | 0.0279 | reports/accepted_bcdo/benchmark_results.csv |
| digits_cnn | bcdo_cnn_reference | neural | 0.6698 | 0.7808 | 49.3972 | 0.0584 | reports/accepted_bcdo/benchmark_results.csv |
| digits_cnn | bcdo_mainline_legacy | neural | 0.9730 | 0.6679 | 49.4188 | 0.0584 | reports/accepted_bcdo/benchmark_results.csv |
| digits_cnn | blockwise_consensus_direction_optimizer | neural | 0.8142 | 0.7403 | 53.8544 | 0.0584 | reports/accepted_bcdo/benchmark_results.csv |
| digits_cnn | rmsprop | neural | 0.2505 | 0.9206 | 6.5982 | 0.0140 | reports/accepted_bcdo/benchmark_results.csv |
| digits_cnn | sgd_momentum | neural | 0.2053 | 0.9342 | 7.4157 | 0.0140 | reports/accepted_bcdo/benchmark_results.csv |
| digits_mlp | adamw | neural | 0.0754 | 0.9779 | 1.6332 | 0.0685 | reports/accepted_bcdo/benchmark_results.csv |
| digits_mlp | bcdo_cnn_reference | neural | 0.0771 | 0.9811 | 37.3843 | 0.1424 | reports/accepted_bcdo/benchmark_results.csv |
| digits_mlp | bcdo_mainline_legacy | neural | 0.0795 | 0.9798 | 38.6555 | 0.1424 | reports/accepted_bcdo/benchmark_results.csv |
| digits_mlp | blockwise_consensus_direction_optimizer | neural | 0.0795 | 0.9798 | 39.0280 | 0.1424 | reports/accepted_bcdo/benchmark_results.csv |
| digits_mlp | rmsprop | neural | 0.0830 | 0.9799 | 1.4523 | 0.0342 | reports/accepted_bcdo/benchmark_results.csv |
| digits_mlp | sgd_momentum | neural | 0.0902 | 0.9792 | 1.3831 | 0.0342 | reports/accepted_bcdo/benchmark_results.csv |
| low_rank_matrix_objective | adamw | structure | 0.4802 | nan | 0.2838 | 0.0011 | reports/accepted_bcdo/benchmark_results.csv |
| low_rank_matrix_objective | bcdo_cnn_reference | structure | 0.0341 | nan | 5.4391 | 0.0022 | reports/accepted_bcdo/benchmark_results.csv |
| low_rank_matrix_objective | bcdo_mainline_legacy | structure | 0.0904 | nan | 5.6752 | 0.0022 | reports/accepted_bcdo/benchmark_results.csv |
| low_rank_matrix_objective | blockwise_consensus_direction_optimizer | structure | 0.0904 | nan | 6.1462 | 0.0022 | reports/accepted_bcdo/benchmark_results.csv |
| low_rank_matrix_objective | rmsprop | structure | 0.1042 | nan | 0.2474 | 0.0006 | reports/accepted_bcdo/benchmark_results.csv |
| low_rank_matrix_objective | sgd_momentum | structure | 0.0095 | nan | 0.2387 | 0.0005 | reports/accepted_bcdo/benchmark_results.csv |
| oscillatory_valley | adamw | stress | -0.0711 | nan | 0.3319 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| oscillatory_valley | bcdo_cnn_reference | stress | -0.0711 | nan | 4.2981 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| oscillatory_valley | bcdo_mainline_legacy | stress | -0.0711 | nan | 4.4723 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| oscillatory_valley | blockwise_consensus_direction_optimizer | stress | -0.0711 | nan | 4.5365 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| oscillatory_valley | rmsprop | stress | -0.0711 | nan | 0.2946 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| oscillatory_valley | sgd_momentum | stress | -0.0711 | nan | 0.2763 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| plateau_escape_objective | adamw | stress | -1.0116 | nan | 0.3918 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| plateau_escape_objective | bcdo_cnn_reference | stress | -1.0500 | nan | 4.6314 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| plateau_escape_objective | bcdo_mainline_legacy | stress | -1.0500 | nan | 4.7609 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| plateau_escape_objective | blockwise_consensus_direction_optimizer | stress | -1.0500 | nan | 4.6800 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| plateau_escape_objective | rmsprop | stress | -1.0500 | nan | 0.3464 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| plateau_escape_objective | sgd_momentum | stress | -1.0500 | nan | 0.3958 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| saddle_objective | adamw | stability | -4.1301 | nan | 0.3159 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| saddle_objective | bcdo_cnn_reference | stability | -4.1322 | nan | 4.3452 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| saddle_objective | bcdo_mainline_legacy | stability | -4.1322 | nan | 4.6936 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| saddle_objective | blockwise_consensus_direction_optimizer | stability | -4.1322 | nan | 4.5589 | 0.0001 | reports/accepted_bcdo/benchmark_results.csv |
| saddle_objective | rmsprop | stability | -4.1322 | nan | 0.2735 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| saddle_objective | sgd_momentum | stability | -4.1322 | nan | 0.2591 | 0.0000 | reports/accepted_bcdo/benchmark_results.csv |
| wine_mlp | adamw | neural | 0.0589 | 0.9852 | 1.3113 | 0.0242 | reports/accepted_bcdo/benchmark_results.csv |
| wine_mlp | bcdo_cnn_reference | neural | 0.0529 | 0.9852 | 32.9175 | 0.0506 | reports/accepted_bcdo/benchmark_results.csv |
| wine_mlp | bcdo_mainline_legacy | neural | 0.0580 | 0.9852 | 35.2551 | 0.0506 | reports/accepted_bcdo/benchmark_results.csv |
| wine_mlp | blockwise_consensus_direction_optimizer | neural | 0.0580 | 0.9852 | 33.9863 | 0.0506 | reports/accepted_bcdo/benchmark_results.csv |
| wine_mlp | rmsprop | neural | 0.0495 | 0.9852 | 1.2069 | 0.0121 | reports/accepted_bcdo/benchmark_results.csv |
| wine_mlp | sgd_momentum | neural | 0.0097 | 1.0000 | 1.0978 | 0.0121 | reports/accepted_bcdo/benchmark_results.csv |