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

The Kaggle version is:

```text
notebooks/lipreading_baseline_sem4_kaggle.ipynb
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

The first baseline showed that a simple 3D-CNN can collapse to predicting the most frequent class. A class-weighted loss and balanced sampling were tested as first fixes for class imbalance, but the model still remained weak.

The current notebooks therefore add a stronger temporal baseline:

```text
mouth crop video
-> frame-level CNN encoder
-> BiGRU temporal encoder
-> word classifier
```

The notebooks also compare RGB input with grayscale input. Grayscale is tested as a possible optimization because lip-reading mostly depends on mouth shape and motion rather than color.

After the speaker-based experiments, the notebooks include a random-split sanity
check for the grayscale CNN+BiGRU model. This is not the main evaluation metric:
it is used to diagnose whether the model can learn the task when speakers are
mixed across train/validation/test. The speaker-based split remains the honest
generalization test.

The notebooks also include a model-based clip audit. It scores clips with the
trained grayscale CNN+BiGRU model and exports high-loss / wrong-prediction CSV
files plus contact-sheet images. This is used to inspect only suspicious clips
instead of manually reviewing the whole dataset.

Before adding more complex architectures blindly, the notebooks now include a
small memorization diagnostic:

```text
Tiny Overfit Test
```

The first run reached 87.5% memorization accuracy on 16 clips and 65.6% on
32 clips. The current stable version fixes all random seeds, decodes the clips
once into memory, uses `num_workers=0`, and replaces batch-dependent BatchNorm
with GroupNorm in a separate diagnostic model. It trains for up to 150 epochs
and stops at 100% memorization.

The diagnostic also checks path/label round trips, conflicting duplicate
tensors, zero-motion clips, and zero-variance clips. It saves training history,
remaining errors, and first/middle/last-frame images for those errors. The gates
are 95% for 16 clips and 90% for 32 clips. Full-dataset model experiments should
continue only after both gates pass.

The current next model step is an LRW-style baseline:

```text
grayscale mouth ROI
-> ResNet18 frame encoder
-> BiGRU temporal encoder
-> word classifier
```

This is implemented with `torchvision.models.resnet18` and does not require
downloading external lip-reading checkpoints. LRW-pretrained weights can be a
separate future integration step.

The notebooks also add a frozen ImageNet-pretrained ResNet18 experiment. In that
setup the ResNet18 frame encoder is frozen and only the BiGRU temporal encoder
and word classifier are trained. This tests whether general pretrained visual
features help on the small coursework dataset.

Visual inspection of `mouth_crops_padded/` also showed that some automatically
created mouth crops contain wrong regions. The next dataset step is manual or
semi-automatic filtering of bad mouth crops and rebuilding the baseline split.

## How to Run the Baseline

### Colab

1. Upload `kaggle_lipreading_dataset.zip` or `colab_lipreading_dataset.zip` to Google Drive.
2. Open `notebooks/lipreading_baseline_sem4_clean.ipynb` in Google Colab.
3. Enable GPU runtime.
4. Run notebook cells from top to bottom.

The notebook restores the dataset archive from Google Drive into Colab temporary
storage. After defining `FrameCNN+BiGRU`, first run the `Tiny Overfit Test`
sections to check whether the model can memorize a very small training subset.
Stop at `Tiny Overfit Decision Gate` if it prints `PIPELINE CHECK`; the heavier
model sections below are intentionally gated by this result.

### Kaggle

1. Upload `kaggle_lipreading_dataset.zip` as a Kaggle Dataset.
2. Create a Kaggle Notebook and add this dataset through **Add Input**.
3. Open/run `notebooks/lipreading_baseline_sem4_kaggle.ipynb`.
4. Enable GPU accelerator.

The Kaggle notebook reads data from `/kaggle/input` and saves checkpoints to
`/kaggle/working/lipreading_sem4`.

## Next Steps

- Run the stable tiny-overfit gates on 16 and 32 clips.
- If either gate fails, use the generated error CSV and frame triplets to debug the pipeline.
- If both gates pass, train models for 30 epochs with early stopping and save the best checkpoint by validation macro-F1.
- Compare grayscale `FrameCNN+BiGRU`, frozen ImageNet `ResNet18+BiGRU`, and then an LRW-style temporal model.
- Keep the speaker-based split as the main evaluation; use random split only as a sanity check.
- Report accuracy, macro-F1, balanced accuracy, per-class recall, prediction distribution, and confusion matrix.
- Freeze the current 955-clip split as dataset v1, then add 2-4 speakers and target weak word classes for dataset v2.
- Repeat the two best models with three random seeds and evaluate the selected model on test once.
