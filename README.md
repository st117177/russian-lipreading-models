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

Dataset v1 is frozen for baseline comparison. Problematic speakers `spk09` and
`spk10` are excluded. Its recommended speaker-based split contains:

- train: 700 clips, speakers `spk03`, `spk04`, `spk05`
- validation: 105 clips, speaker `spk06`
- test: 150 clips, speakers `spk07`, `spk08`
- total: 955 clips, 10 words, 6 speakers

Dataset v2 adds two new anonymized speakers to train while preserving the v1
validation and test sets:

- source material: 5 videos, about 147 minutes, split into 32 WebMAUS chunks
- extracted: 437 word-level clips
- after automatic checks: 408 mouth crops
- after contact-sheet review: 400 clean new mouth crops
- train: 1100 clips, speakers `spk03`, `spk04`, `spk05`, `spk11`, `spk12`
- validation: 105 clips, speaker `spk06` (unchanged from v1)
- test: 150 clips, speakers `spk07`, `spk08` (unchanged from v1)
- total: 1355 clips, 10 words, 8 speakers

Source video URLs, channel names, and original video identifiers are private
local metadata and are intentionally not stored in this repository.

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

The dataset v1 split files are in:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\03_splits\ml_splits_mouth_crops_padded_clean_manual_no_spk09_10\speaker_top10
```

The dataset v2 split files are in:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\03_splits\ml_splits_mouth_crops_padded_dataset_v2\speaker_top10
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
- `package_ml_dataset.py`

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

Visual inspection showed that automatic face detection can occasionally crop a
wrong region. Dataset v2 therefore uses automatic checks followed by contact-sheet
review of suspicious crops. Eight bad new crops were removed through a local
rejection list without deleting or publishing source metadata.

## How to Run the Baseline

The local preprocessing scripts use a separate Python environment. Install the
reproducible dependencies with:

```text
python -m pip install -r requirements-pipeline.txt
```

### Colab

For the current diagnostic step, use the short standalone notebook:

```text
notebooks/lipreading_tiny_overfit_colab.ipynb
```

It contains only dataset restoration and the stable 16/32-clip tiny-overfit
checks, so it does not rerun the older baseline or ResNet experiments.

After both tiny-overfit gates pass, use the controlled full-dataset baseline:

```text
notebooks/lipreading_framecnn_bigru_controlled_colab.ipynb
```

This notebook trains the grayscale `FrameCNN+BiGRU` for at most 30 epochs on
the speaker-based split, stops early after seven epochs without improvement,
and selects the best checkpoint by validation macro-F1. It saves the checkpoint,
history, per-class report, validation predictions, curves, and confusion matrix
to Google Drive. The test split is deliberately not evaluated.

The next controlled transfer-learning experiment is:

```text
notebooks/lipreading_frozen_resnet18_bigru_colab.ipynb
```

It extracts ImageNet-pretrained ResNet18 frame features once, caches them on
Google Drive, and trains only a BiGRU word classifier. The frozen encoder is kept
in evaluation mode, grayscale frames are repeated to three channels and
ImageNet-normalized, and the test split remains unused.

The follow-up imbalance diagnostic reuses the same cached features:

```text
notebooks/lipreading_frozen_resnet18_bigru_balanced_colab.ipynb
```

It changes only the training sampler. Every non-empty `(speaker, word)` group
receives equal total sampling probability, which reduces the dominance of
`spk04` and frequent words without changing validation or using test data.

This balanced experiment has been completed. It reached validation macro-F1
`0.1669`, balanced accuracy `0.1768`, accuracy `0.2381`, and predicted 9 of 10
classes. This is the best class coverage and macro-F1 so far, but train accuracy
reached about `0.84` while validation remained weak. The current bottleneck is
therefore generalization to an unseen speaker, not a broken training loop.

The controlled dataset-v2 repetition is:

```text
notebooks/lipreading_frozen_resnet18_bigru_balanced_v2_colab.ipynb
```

It is self-contained: it restores dataset v2, extracts and caches new frozen
ResNet18 features, trains the same BiGRU with the same speaker-word balanced
sampler, and compares the resulting validation metrics with the saved dataset-v1
summary. It writes to a separate Drive directory and does not evaluate test.

The local dataset-v2 archive is:

```text
ru_dataset/05_colab_package/colab_lipreading_dataset_v2.zip
```

It contains only the 1355 clips referenced by the v2 split plus split metadata.
The archive is kept outside GitHub.

The first dataset-v2 run completed at validation accuracy `0.1524`, macro-F1
`0.1301`, balanced accuracy `0.1655`, and predictions across all 10 classes.
The corresponding single-seed v1 macro-F1 was `0.1669`, so dataset v2 did not
improve this model in one run. A short cached-feature multi-seed comparison is
provided in:

```text
notebooks/lipreading_v1_v2_balanced_multiseed_colab.ipynb
```

It runs the same balanced BiGRU protocol for v1 and v2 with seeds 42, 123, and
2026, then saves mean/std metrics and paired v2-minus-v1 macro-F1 differences.
It does not decode videos, rerun ResNet18, or use the test split.

The completed comparison gives mean macro-F1 `0.1311 ± 0.0312` for v1 and
`0.1416 ± 0.0458` for v2. Dataset v2 wins on two of three paired seeds and has
higher, more stable balanced accuracy (`0.1715 ± 0.0104` versus
`0.1471 ± 0.0258`). This is a modest positive trend, while absolute validation
quality remains weak.

The next controlled temporal-backend experiment is:

```text
notebooks/lipreading_v2_tcn_backend_multiseed_colab.ipynb
```

It keeps dataset v2, cached frozen ResNet18 features, balanced sampling, and the
same three seeds, but replaces BiGRU with a compact residual TCN. This isolates
the temporal backend; it is not presented as a full LRW-style model because it
does not yet include a lip-reading-pretrained 3D visual frontend.

For the immediate next step, upload/open the TCN notebook in Colab and run it
top to bottom. The required dataset-v2 feature cache is already on Google Drive,
so no dataset ZIP is needed.

The older end-to-end baseline instructions are retained below for reproducibility:

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

- Keep the current 955-clip dataset and its split frozen as dataset v1.
- Keep the prepared 1355-clip dataset and its split as dataset v2.
- Treat the cached-feature multi-seed v1/v2 comparison as complete.
- Run the compact TCN versus BiGRU temporal-backend comparison on dataset v2.
- Then decide whether to integrate a lip-reading-pretrained 3D visual frontend.
- Run the final two variants with three seeds and evaluate the selected model on test once.

The detailed collection and evaluation plan is in `data/DATASET_V2_PLAN.md`.
