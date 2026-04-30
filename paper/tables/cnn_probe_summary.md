# Cnn Probe Summary

Focused CNN probe used to keep the CNN gap visible.

Source CSVs:
- `reports/bcdo_cnn_probe/benchmark_results.csv`

| task | optimizer | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | source_csv |
| --- | --- | --- | --- | --- | --- |
| breast_cancer_mlp | adamw | 0.0614 | 0.9857 | 1.0707 | reports/bcdo_cnn_probe/benchmark_results.csv |
| breast_cancer_mlp | blockwise_consensus_direction_optimizer | 0.0623 | 0.9844 | 26.7326 | reports/bcdo_cnn_probe/benchmark_results.csv |
| breast_cancer_mlp | rmsprop | 0.0567 | 0.9883 | 0.9276 | reports/bcdo_cnn_probe/benchmark_results.csv |
| breast_cancer_mlp | sgd_momentum | 0.0617 | 0.9831 | 0.8475 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_cnn | adamw | 0.3739 | 0.8863 | 4.7015 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_cnn | blockwise_consensus_direction_optimizer | 0.7047 | 0.7577 | 33.4702 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_cnn | rmsprop | 0.2505 | 0.9206 | 4.4834 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_cnn | sgd_momentum | 0.7213 | 0.7577 | 4.4087 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_mlp | adamw | 0.0756 | 0.9779 | 1.2855 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_mlp | blockwise_consensus_direction_optimizer | 0.0774 | 0.9811 | 25.7786 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_mlp | rmsprop | 0.1042 | 0.9824 | 1.1181 | reports/bcdo_cnn_probe/benchmark_results.csv |
| digits_mlp | sgd_momentum | 0.0902 | 0.9792 | 1.0370 | reports/bcdo_cnn_probe/benchmark_results.csv |
| moons_mlp | adamw | 0.2177 | 0.9074 | 0.9916 | reports/bcdo_cnn_probe/benchmark_results.csv |
| moons_mlp | blockwise_consensus_direction_optimizer | 0.1820 | 0.9289 | 23.5298 | reports/bcdo_cnn_probe/benchmark_results.csv |
| moons_mlp | rmsprop | 0.0511 | 0.9948 | 0.8428 | reports/bcdo_cnn_probe/benchmark_results.csv |
| moons_mlp | sgd_momentum | 0.0811 | 0.9810 | 0.7977 | reports/bcdo_cnn_probe/benchmark_results.csv |
| wine_mlp | adamw | 0.1136 | 0.9778 | 1.0459 | reports/bcdo_cnn_probe/benchmark_results.csv |
| wine_mlp | blockwise_consensus_direction_optimizer | 0.0580 | 0.9852 | 22.5779 | reports/bcdo_cnn_probe/benchmark_results.csv |
| wine_mlp | rmsprop | 0.0453 | 0.9852 | 0.9117 | reports/bcdo_cnn_probe/benchmark_results.csv |
| wine_mlp | sgd_momentum | 0.0558 | 0.9852 | 0.8452 | reports/bcdo_cnn_probe/benchmark_results.csv |