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

Dataset v3 is a train-expansion experiment with four additional anonymized
speakers and landmark-aligned mouth crops. It preserves the same validation
speaker and deliberately contains no test split during model selection:

- source material: 4 videos, 200 minutes, split into 40 WebMAUS chunks
- extracted: 817 padded word-level clips
- after automatic quality checks: 816 clips
- after landmark mouth cropping: 816 model inputs
- train: 1901 clips, 9 speakers
- validation: 105 clips, speaker `spk06` (unchanged)
- test: not packaged or loaded during this experiment
- largest train speaker: 472 clips (`24.8%` of train)

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

The dataset v3 train/validation split files are in:

```text
C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\03_splits\ml_splits_mouth_crops_landmark_dataset_v3\speaker_top10
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

The next controlled preprocessing experiment is `landmark v3`. The original
heuristic crop detects a face on a few frames and reuses one fixed lower-face box
for the whole clip. The new script uses per-frame MediaPipe Face Mesh landmarks,
eye-line alignment, temporal smoothing, and a minimum landmark-coverage gate.
It creates a separate model input folder and does not overwrite dataset v2.

The local development split for this experiment contains 1085 train clips and
the unchanged 105-clip `spk06` validation set. Fifteen train clips failed the
70% landmark-coverage threshold; all validation clips passed. Test is omitted
because the previous test split has already been viewed and must not be reused
for tuning.

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

To compare the old heuristic ROI with landmark-aligned preprocessing while
keeping the LRW-pretrained model and training protocol fixed, use:

```text
notebooks/lipreading_lrw_landmark_v3_comparison_colab.ipynb
```

Upload the local archive `colab_lipreading_dataset_landmark_v3.zip` to Google
Drive first. The decision is based on validation macro-F1 over seeds 42, 123,
and 2026; the old-crop reference is 0.2736 +/- 0.0290.

The landmark experiment completed successfully. Mean validation results are
accuracy `0.5460 +/- 0.0110`, macro-F1 `0.5359 +/- 0.0139`, and balanced
accuracy `0.5787 +/- 0.0108`. It predicts all 10 classes and improves mean
macro-F1 by `+0.2622` over the same LRW-pretrained model with the old crop.
The landmark version wins on all three paired seeds and is now the selected
preprocessing pipeline.

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

The completed TCN comparison gives mean macro-F1 `0.1182 +/- 0.0287`, compared
with `0.1416 +/- 0.0458` for BiGRU. TCN wins on one of three paired seeds. Its
mean balanced accuracy is similar (`0.1742` versus `0.1715`), but accuracy and
class coverage are lower. Therefore changing only the temporal backend does not
solve the generalization bottleneck.

The next controlled visual-pretraining experiment is:

```text
notebooks/lipreading_lrw_pretrained_frontend_bigru_multiseed_colab.ipynb
```

It replaces the frozen ImageNet frame encoder with an official LRW-pretrained
`3D Conv + ResNet18` visual frontend, while retaining the same dataset v2,
balanced sampler, BiGRU classifier, seeds, validation split, and metrics. The
pretrained model is used for non-commercial coursework research; its upstream
repository and license are linked inside the notebook. Test remains untouched.

The LRW-pretrained experiment completed successfully. Mean validation metrics
over seeds 42, 123, and 2026 are accuracy `0.2857 +/- 0.0530`, macro-F1
`0.2736 +/- 0.0290`, and balanced accuracy `0.3157 +/- 0.0039`. The previous
ImageNet frontend reached mean macro-F1 `0.1416`; LRW pretraining improves it by
`0.1320` and wins on all three paired seeds. This confirms visual-speech
pretraining as the most important model change tested in this project.

Before the landmark experiment, the old-crop checkpoints were evaluated without
further training in:

```text
notebooks/lipreading_lrw_pretrained_final_test_colab.ipynb
```

This notebook loads the three already selected validation checkpoints, extracts
LRW features for the reserved test speakers once, and reports the historical
old-crop test metrics. Test results must not be used for model selection.

That historical test-only evaluation is complete. Across the three old-crop
checkpoints, test accuracy is `0.1822 +/- 0.0278`, macro-F1 is
`0.1530 +/- 0.0206`, and balanced accuracy is `0.1804 +/- 0.0250`. The model
predicts an average of `9.67/10` classes. Test macro-F1 is lower than validation
macro-F1 (`0.2736`), showing that generalization to the two reserved speakers
was the main limitation of the old preprocessing. Those speakers have now been
viewed and are not reused as an untouched final test for landmark v3.

A compact, report-ready summary is available in
`reports/FINAL_MODEL_RESULTS.md`.

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
- Treat the compact TCN versus BiGRU comparison as complete: TCN did not improve
  mean validation macro-F1.
- Treat the LRW-pretrained visual-frontend comparison as complete: it wins on
  all three seeds.
- Treat landmark v3 versus the old heuristic crop as complete: landmark v3
  improves validation macro-F1 from `0.2736` to `0.5359` and wins on all seeds.
- Freeze the LRW-pretrained + landmark + balanced BiGRU configuration. Do not
  tune it further on `spk06` or the previously viewed `spk07`/`spk08` test.
- Treat the new 508-clip, three-speaker landmark-v3 final test as frozen. The
  one-time three-seed evaluation is complete: `0.4370 +/- 0.0104` accuracy,
  `0.3919 +/- 0.0158` macro-F1, and `0.4012 +/- 0.0179` balanced accuracy.
- Keep `notebooks/lipreading_lrw_landmark_v3_untouched_final_test_colab.ipynb`
  as the reproducible test-only evaluation; do not tune on its results.
- Treat dataset v3 preprocessing as complete: four new train speakers produced
  816 landmark crops, giving 1901 train clips while keeping `spk06` validation
  unchanged.
- Treat the dataset v3 all-speaker comparison as complete. With the fixed
  LRW-pretrained + balanced BiGRU protocol, dataset v3 reached validation
  macro-F1 `0.5091 +/- 0.0214` versus `0.5359 +/- 0.0139` for dataset v2 and
  lost on all three paired seeds.
- Treat the dataset v3 single-speaker GPU ablation as complete. The unchanged
  dataset v2 control reproduced `0.5359 +/- 0.0139` macro-F1. No new speaker
  improved the mean: `spk16` was closest at `0.5337 +/- 0.0440` and won two of
  three paired seeds, while `spk17`, `spk18`, and `spk19` reduced the mean.
- Treat the dataset v3 sampler ablation as complete. `uniform_shuffle` reached
  validation macro-F1 `0.5510 +/- 0.0081`, compared with `0.5091 +/- 0.0214`
  for the previous `speaker_word_inverse` sampler. It improved every paired
  seed, with a mean delta of `+0.0420`. The simpler shuffle also avoids heavily
  repeating speaker-word groups that contain only one to four clips.
- Use `uniform_shuffle` for the next dataset v3 training run. The sampler
  result is validation-only and does not replace the already frozen final-test
  result.
- Treat the `ResNet18.layer4` fine-tuning gate as complete. The best seed-42
  validation macro-F1 was `0.5545` versus `0.5454` for the frozen baseline, a
  gain of only `+0.0092`; train macro-F1 approached `1.0`, so the encoder remains
  frozen.
- Treat frozen horizontal-flip augmentation as a positive validation result.
  `notebooks/lipreading_lrw_frozen_hflip_dataset_v3_colab.ipynb` reached mean
  validation macro-F1 `0.5775 +/- 0.0095`, improved all three paired seeds, and
  produced a mean paired gain of `+0.0264` over `uniform_shuffle`.
- Keep horizontal flip for the next candidate training run, but do not reuse the
  already viewed final test to tune or select this candidate.
- Treat the cross-speaker hflip check as complete. Holding out `spk03`, `spk17`,
  and `spk19` in turn, hflip improved mean macro-F1 from `0.3727` to `0.3900`,
  a paired gain of `+0.0173`, and won 6 of 9 fold-seed comparisons. The effect
  is positive but noisy, so it remains a modest augmentation rather than the
  main source of model quality.
- Use `notebooks/lipreading_lrw_frozen_hflip_cross_speaker_colab.ipynb` to
  reproduce this check. It never loads the final test split.
- Treat the official LRW-pretrained MS-TCN transfer experiment as complete.
  Across held-out `spk03`, `spk17`, and `spk19` and three seeds, the frozen
  pretrained MS-TCN plus a new linear classifier reached mean macro-F1
  `0.5832`, versus `0.3900` for the hflip BiGRU. The mean paired gain was
  `+0.1932`, and MS-TCN won 8 of 9 comparisons.
- Use
  `notebooks/lipreading_lrw_pretrained_mstcn_transfer_cross_speaker_colab.ipynb`
  to reproduce the comparison. It uses the official pretrained temporal
  backend and never loads the final test split.
- Keep the pretrained MS-TCN as the leading validation candidate. The next
  decision should be whether to freeze this result for the coursework or
  collect a genuinely new untouched test before one final evaluation.
- Treat temporal masking for the pretrained MS-TCN as a completed negative
  ablation. Masking four consecutive frames in 50% of train examples reached
  mean validation macro-F1 `0.5391` versus `0.5832` without masking, with a
  paired delta of `-0.0441` and `0/9` wins. Keep the simpler unmasked MS-TCN.
- Use
  `notebooks/lipreading_lrw_pretrained_mstcn_temporal_masking_cross_speaker_colab.ipynb`
  and `reports/results/dataset_v3_pretrained_mstcn_temporal_masking_*.csv` to
  reproduce this validation-only ablation. It never loads the final test split.
- Treat the official pretrained DC-TCN transfer experiment as complete. It
  reached mean validation macro-F1 `0.2834` versus `0.3900` for hflip BiGRU,
  lost all 9 paired comparisons, and is rejected by the decision gate.
- Use
  `notebooks/lipreading_lrw_pretrained_dctcn_transfer_cross_speaker_colab.ipynb`
  and the two `reports/results/dataset_v3_pretrained_dctcn_transfer_*.csv`
  files to reproduce and inspect this validation-only result.
- Keep test data closed until the dataset v3 validation decision is recorded.
- Use `reports/FINAL_MODEL_RESULTS.md` for the coursework results section.
- P0 data audit is recorded in `reports/P0_DATA_AUDIT.md`. The prepared
  `notebooks/lipreading_lrw_pretrained_mstcn_posthoc_final_test_colab.ipynb`
  is the next P1 step; it evaluates MS-TCN on the existing final-test only as a
  post-hoc benchmark, because that test was previously used for BiGRU.

The detailed collection and evaluation plan is in `data/DATASET_V2_PLAN.md`.
