# Accepted Vs Runnable Summary

Separation between the fixed accepted public snapshot and the current writable runnable output path.

Source CSVs:
- `reports/accepted_bcdo/benchmark_results.csv`
- `reports/bcdo_mainline/benchmark_results.csv`

| task | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | line | source_csv |
| --- | --- | --- | --- | --- | --- |
| block_structure_classification | 0.3618 | 0.8455 | 38.5925 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| block_structure_classification | 0.3618 | 0.8455 | 38.5925 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| breast_cancer_mlp | 0.0617 | 0.9857 | 39.5980 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| breast_cancer_mlp | 0.0617 | 0.9857 | 39.5980 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| digits_cnn | 0.8142 | 0.7403 | 53.8544 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| digits_cnn | 0.8142 | 0.7403 | 53.8544 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| digits_mlp | 0.0795 | 0.9798 | 39.0280 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| digits_mlp | 0.0795 | 0.9798 | 39.0280 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| direction_reversal_objective | 0.0017 | nan | 4.6071 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| direction_reversal_objective | 0.0017 | nan | 4.6071 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| low_rank_matrix_objective | 0.0904 | nan | 6.1462 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| low_rank_matrix_objective | 0.0904 | nan | 6.1462 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| moons_mlp | 0.1820 | 0.9289 | 35.2892 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| moons_mlp | 0.1820 | 0.9289 | 35.2892 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| oscillatory_valley | -0.0711 | nan | 4.5365 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| oscillatory_valley | -0.0711 | nan | 4.5365 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| plateau_escape_objective | -1.0500 | nan | 4.6800 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| plateau_escape_objective | -1.0500 | nan | 4.6800 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| saddle_objective | -4.1322 | nan | 4.5589 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| saddle_objective | -4.1322 | nan | 4.5589 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| wine_mlp | 0.0580 | 0.9852 | 33.9863 | accepted_public_snapshot | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |
| wine_mlp | 0.0580 | 0.9852 | 33.9863 | current_runnable_output | reports/accepted_bcdo/benchmark_results.csv | reports/bcdo_mainline/benchmark_results.csv |