# Data Metadata

This folder contains small metadata files used to reproduce the semester 4 dataset
pipeline.

Current contents:

- `DATASET_V1.md` - frozen statistics and evaluation rules for the current
  955-clip coursework baseline.
- `DATASET_V2_PLAN.md` - concrete collection, split, and comparison plan for
  expanding the dataset with new speakers.

The private source manifest containing creator names and video URLs is kept
locally outside this repository. It is intentionally not published on GitHub.

The generated video dataset is stored outside this GitHub repository because it
contains downloaded videos, generated clips, mouth crops, and Colab archives.

Local dataset location used during development:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset
```

Current local training split:

```text
03_splits/ml_splits_mouth_crops_padded_clean_manual_no_spk09_10/speaker_top10/
```

Current local video root for those split CSV files:

```text
02_model_inputs/
```
