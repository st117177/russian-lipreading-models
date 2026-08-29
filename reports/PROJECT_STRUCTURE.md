# Project Structure

This file explains the intended structure of the semester 4 project. It separates
local generated data from the GitHub repository.

## Recommended Local Layout

```text
Lip-reading-demo-dataset/
├── semester3_coursework/
│   ├── lip-read-demo-dataset/
│   ├── archives/
│   │   ├── lip-read-demo-dataset.zip
│   │   └── lip-read-coursework-materials.zip
│   └── Tarasova_lip_read_rus_dataset_report (3) (1).pdf
│
└── semester4_coursework/
    ├── dataset_local/
    │   ├── private_metadata/
    │   │   └── candidate_videos.csv
    │   └── ru_dataset/
    │       ├── 00_raw/
    │       ├── 01_intermediate_clips/
    │       ├── 02_model_inputs/
    │       ├── 03_splits/
    │       ├── 04_quality_reports/
    │       ├── 05_colab_package/
    │       ├── 06_logs/
    │       └── 99_archive_old_versions/
    │
    ├── github_repo/
    │   └── russian-lipreading-sem4-coursework/
    │       ├── scripts/
    │       ├── notebooks/
    │       ├── reports/
    │       ├── data/
    │       └── README.md
    │
    └── notes/
        ├── SEMESTER4_STATUS.md
        └── README_PIPELINE.md
```

## Why Dataset And Repository Are Separate

The generated dataset contains video files and intermediate artifacts. It is too
large and too noisy to store directly in GitHub.

The GitHub repository should contain:

- scripts;
- notebooks;
- reports;
- small metadata files;
- instructions for reproducing the dataset.

The local dataset folder should contain:

- the private source manifest with creator names and video URLs;
- downloaded videos;
- WebMAUS chunks;
- generated word clips;
- mouth crops;
- split CSV files;
- Colab archive.

`notes/_archive_not_for_work/` may contain old duplicate folders, but it is not part
of the normal workflow.

## Current Important Dataset Files

Use these files for the current baseline:

```text
semester4_coursework/dataset_local/ru_dataset/02_model_inputs/mouth_crops_padded/
semester4_coursework/dataset_local/ru_dataset/02_model_inputs/mouth_crops_padded_labels.csv
semester4_coursework/dataset_local/ru_dataset/03_splits/ml_splits_mouth_crops_padded_clean_manual_no_spk09_10/speaker_top10/
semester4_coursework/dataset_local/ru_dataset/05_colab_package/colab_lipreading_dataset.zip
```

Older folders without `_padded` are kept only for comparison. The non-padded clips
were cut too tightly and should not be used as the main training input.
