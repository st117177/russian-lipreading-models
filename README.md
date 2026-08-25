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

## Repository Structure

```text
scripts/      dataset preparation, WebMAUS, clipping, quality check, split, crops
notebooks/    Colab experiments and PyTorch baseline drafts
reports/      status reports and coursework notes
data/         small metadata files, not full video datasets
```

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
