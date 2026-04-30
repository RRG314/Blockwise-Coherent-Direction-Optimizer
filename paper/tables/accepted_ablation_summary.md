# Accepted Ablation Summary

Accepted public ablation summary for the mainline BCDO design.

Source CSVs:
- `reports/accepted_bcdo/ablation_results.csv`

| variant_name | mean_best_val_loss | mean_best_val_accuracy | mean_runtime_per_step_ms | mean_selection_score | source_csv |
| --- | --- | --- | --- | --- | --- |
| rmsprop_baseline | 0.2578 | 0.9084 | 1.8813 | 0.3088 | reports/accepted_bcdo/ablation_results.csv |
| v42_baseline | 0.5167 | 0.7436 | 25.2511 | 0.1932 | reports/accepted_bcdo/ablation_results.csv |
| recoverability_periodic | 0.5576 | 0.7300 | 24.1702 | 0.1635 | reports/accepted_bcdo/ablation_results.csv |
| no_conv_fallback_relaxation | 0.5734 | 0.7240 | 25.3669 | 0.1505 | reports/accepted_bcdo/ablation_results.csv |
| no_conv_support_bonus | 0.5734 | 0.7240 | 25.9895 | 0.1505 | reports/accepted_bcdo/ablation_results.csv |
| v4_fast_full | 0.5734 | 0.7240 | 25.8552 | 0.1505 | reports/accepted_bcdo/ablation_results.csv |
| no_trusted_direction | 0.5874 | 0.7087 | 23.9213 | 0.1416 | reports/accepted_bcdo/ablation_results.csv |
| no_stable_consensus | 0.5919 | 0.7121 | 22.8397 | 0.1368 | reports/accepted_bcdo/ablation_results.csv |
| no_conv_structure_support | 0.5981 | 0.7004 | 25.2492 | 0.1367 | reports/accepted_bcdo/ablation_results.csv |
| no_conv_profile_split | 0.6053 | 0.6946 | 25.4671 | 0.1333 | reports/accepted_bcdo/ablation_results.csv |
| no_matrix_consensus | 0.5932 | 0.7072 | 21.4873 | 0.1284 | reports/accepted_bcdo/ablation_results.csv |
| v4_fast_legacy | 0.6174 | 0.6814 | 25.3133 | 0.1259 | reports/accepted_bcdo/ablation_results.csv |
| no_block_structure | 0.6323 | 0.6800 | 22.3284 | 0.1234 | reports/accepted_bcdo/ablation_results.csv |
| adamw_baseline | 0.8389 | 0.7819 | 2.0591 | -0.1641 | reports/accepted_bcdo/ablation_results.csv |