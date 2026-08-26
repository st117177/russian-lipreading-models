# Data Metadata

This folder contains small metadata files used to reproduce the semester 4 dataset
pipeline.

Current contents:

- `candidate_videos.csv` - candidate YouTube videos grouped by speaker.

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
