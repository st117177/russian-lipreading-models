# Russian Lip Reading Coursework, Semester 4

This repository contains the code and notes for extending a Russian lip-reading coursework project from a small demo dataset to a larger dataset and a PyTorch baseline model.

## Current Status

The previous semester demo dataset contained:

- 182 clips
- 8 words
- 5 speakers

The expanded clean padded dataset currently contains:

- 1222 clean clips
- 14 words
- 8 speakers

For the current baseline, problematic speakers `spk09` and `spk10` are excluded. The recommended speaker-based split contains:

- train: 700 clips, speakers `spk03`, `spk04`, `spk05`
- validation: 105 clips, speaker `spk06`
- test: 150 clips, speakers `spk07`, `spk08`
- total: 955 clips, 10 words, 6 speakers

The full dataset is not stored in this repository because it contains generated video files. It should be stored separately, for example on Google Drive.

In the local workspace used for this coursework, the generated dataset is stored at:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset
```

The local dataset is organized into numbered folders. The current local model input
videos are in:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\02_model_inputs\mouth_crops_padded
```

The current local split files are in:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\03_splits\ml_splits_mouth_crops_padded_clean_manual_no_spk09_10\speaker_top10
```

## Repository Structure

```text
scripts/      dataset preparation, WebMAUS, clipping, quality check, split, crops
notebooks/    Colab experiments and PyTorch baseline drafts
reports/      status reports and coursework notes
data/         small metadata files, not full video datasets
```

More detailed structure notes:

```text
reports/PROJECT_STRUCTURE.md
scripts/README.md
reports/MOUTH_CROP_CLEANING_PLAN.md
```

## Script Status

Detailed script descriptions are in:

```text
scripts/README.md
```

The scripts are split into three groups:

### Kept From Semester 3

These scripts are the original foundation of the coursework pipeline:

- `batch_prepare_maus_inputs.py`
- `submit_webmaus_basic.py`
- `batch_submit_webmaus_basic.py`
- `get_phonewords_frames.py`
- `batch_generate_phonewords_frames.py`
- `cut_ru_clips_from_words_frames.py`
- `validate_ru_dataset.py`

### Edited For Semester 4

These scripts existed before, but were extended for the larger automated dataset:

- `prepare_maus_inputs.py` - transcript/SRT cleanup and WebMAUS input preparation.
- `batch_cut_ru_clips_from_words_frames.py` - batch word clipping with padded timing.

### Added In Semester 4

These scripts were added for the dataset expansion, quality control, ML splits, and
manual crop cleaning:

- `download_candidate_videos.py`
- `chunk_raw_videos_for_maus.py`
- `build_labels_from_clips.py`
- `quality_check_clips.py`
- `create_mouth_crops.py`
- `apply_manual_mouth_crop_filter.py`
- `create_ml_splits.py`

## Current Baseline

The recommended clean Colab notebook is:

```text
notebooks/lipreading_baseline_sem4_clean.ipynb
```

The original exploratory draft is kept separately:

```text
notebooks/lipreading_baseline_sem4_draft.ipynb
```

The baseline pipeline:

```text
CSV labels
-> OpenCV video loading
-> tensor [3, 24, 96, 96]
-> PyTorch Dataset
-> DataLoader
-> Simple3DCNN
-> CrossEntropyLoss
-> train/validation loop
```

The first baseline showed that a simple 3D-CNN can collapse to predicting the most frequent class. A class-weighted loss was tested as a first fix for class imbalance, but more work is needed.

Visual inspection of `mouth_crops_padded/` also showed that some automatically
created mouth crops contain wrong regions. The next dataset step is manual or
semi-automatic filtering of bad mouth crops and rebuilding the baseline split.

## How to Run the Baseline

1. Upload `colab_lipreading_dataset.zip` to Google Drive.
2. Open `notebooks/lipreading_baseline_sem4_clean.ipynb` in Google Colab.
3. Enable GPU runtime.
4. Run notebook cells from top to bottom.

The notebook restores the dataset archive from Google Drive into Colab temporary storage and trains a simple 3D-CNN baseline.

## Next Steps

- Replace or exclude low-quality speakers.
- Improve class balance.
- Add balanced sampling or oversampling.
- Train a stronger baseline.
- Add plots for loss/accuracy.
- Add confusion matrix and per-class accuracy.
- Write the semester 4 report.
