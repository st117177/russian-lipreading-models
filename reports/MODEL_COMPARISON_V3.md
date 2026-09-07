# Model comparison: dataset v3

## Purpose

This report collects the experiments already run on dataset v3. The main
comparison uses the same cross-speaker protocol: three held-out speakers and
three seeds per fold. The final test is not included here because it is
reserved for the post-hoc MS-TCN benchmark and was already viewed in an older
BiGRU experiment.

## Main comparison

| Model or condition | Protocol | Validation accuracy | Validation macro-F1 | Decision |
|---|---|---:|---:|---|
| Pretrained LRW ResNet18 + MS-TCN | 3 speaker folds x 3 seeds | 0.5974 +/- 0.0815 | 0.5832 +/- 0.0785 | Keep |
| Pretrained LRW ResNet18 + MS-TCN + temporal masking | same folds and seeds | 0.5691 +/- 0.0876 | 0.5391 +/- 0.0656 | Reject |
| Pretrained LRW ResNet18 + DC-TCN | same folds and seeds | 0.3152 +/- 0.0443 | 0.2834 +/- 0.0497 | Reject |
| BiGRU with horizontal flip | same cross-speaker comparison | not used for decision | 0.3900 mean | Control |

The MS-TCN advantage over the paired BiGRU control is `+0.1932` macro-F1 on
average across nine paired comparisons. Temporal masking loses in all nine
paired comparisons, so it is retained only as a negative ablation.

## Results with a different protocol

The separate v3 multiseed experiment with one validation speaker produced:

- LRW-pretrained frontend + BiGRU: accuracy `0.5302`, macro-F1 `0.5091 +/- 0.0214`;
- v2 under the same single-split style: macro-F1 `0.5359`;
- v3 under that comparison: macro-F1 `0.5091`.

These numbers are useful for diagnosing split and dataset-version effects, but
must not be merged into the three-fold cross-speaker table. The validation
speaker sets, training protocol and baseline construction differ.

## Current decision

1. Keep pretrained MS-TCN as the leading architecture for the coursework.
2. Do not spend more compute on DC-TCN or temporal masking.
3. Complete the padded-boundary review before using the final-test notebook.
4. Run the fixed P1 post-hoc notebook once on the existing 508-clip final test.
5. Only if P1 is informative, consider one small adapter experiment. It must
   be selected using validation data and must not tune on the post-hoc test.

## Reproducibility

Source files for the main rows are:

- `reports/results/dataset_v3_pretrained_mstcn_cross_speaker_runs.csv`;
- `reports/results/dataset_v3_pretrained_mstcn_temporal_masking_runs.csv`;
- `reports/results/dataset_v3_pretrained_dctcn_transfer_cross_speaker_runs.csv`;
- `reports/results/dataset_v3_pretrained_mstcn_cross_speaker_summary.csv`.

The next result will be written by
`notebooks/lipreading_lrw_pretrained_mstcn_posthoc_final_test_colab.ipynb`.
