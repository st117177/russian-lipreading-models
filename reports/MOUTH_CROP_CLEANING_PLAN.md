# Mouth Crop Cleaning Plan

## Problem

The automatic mouth-crop stage creates videos in:

```text
semester4_coursework/dataset_local/ru_dataset/02_model_inputs/mouth_crops_padded/
```

Some crops are wrong: instead of the mouth area, they may contain background, neck,
only part of the face, or an unstable crop. These clips can remain in the training
split and make the baseline model learn poorly.

## Why This Happened

`quality_check_clips.py` checked the generated word clips automatically, mostly by
readability, duration, brightness, and face detection.

`create_mouth_crops.py` then created mouth crops. If face detection was uncertain,
the script could use a fallback crop. This fallback sometimes selected a wrong
region.

So the current labels mean "the file exists", not "the mouth crop is visually good".

## Do Not Delete Videos Manually

Do not remove `.mp4` files directly from `mouth_crops_padded/`. If files are deleted
without updating CSV labels, the dataset and split files become inconsistent.

Instead, keep a manual rejection list and generate a new clean labels file from it.

## Manual Cleaning Workflow

1. Open folders inside `semester4_coursework/dataset_local/ru_dataset/02_model_inputs/mouth_crops_padded/`.
2. For every bad crop, copy its relative path, for example:

```text
mouth_crops_padded/время/spk04/source_spk04_01_part001_000394.mp4
```

3. Add it to:

```text
semester4_coursework/dataset_local/ru_dataset/02_model_inputs/manual_bad_mouth_crops.csv
```

with a reason:

```csv
clip_path,reason,notes
mouth_crops_padded/время/spk04/source_spk04_01_part001_000394.mp4,wrong_roi,bookshelf instead of mouth
```

4. Generate a new labels file without those clips:

```text
semester4_coursework/dataset_local/ru_dataset/02_model_inputs/mouth_crops_padded_labels_clean_manual.csv
```

Command:

```powershell
cd C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset
python semester4_coursework\github_repo\russian-lipreading-sem4-coursework\scripts\apply_manual_mouth_crop_filter.py `
  --labels semester4_coursework\dataset_local\ru_dataset\02_model_inputs\mouth_crops_padded_labels.csv `
  --bad-list semester4_coursework\dataset_local\ru_dataset\02_model_inputs\manual_bad_mouth_crops.csv `
  --out-labels semester4_coursework\dataset_local\ru_dataset\02_model_inputs\mouth_crops_padded_labels_clean_manual.csv
```

5. Rebuild train/validation/test splits from the manually cleaned labels.

Command:

```powershell
cd C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset
python semester4_coursework\github_repo\russian-lipreading-sem4-coursework\scripts\create_ml_splits.py `
  --labels semester4_coursework\dataset_local\ru_dataset\02_model_inputs\mouth_crops_padded_labels_clean_manual.csv `
  --out-dir semester4_coursework\dataset_local\ru_dataset\03_splits\ml_splits_mouth_crops_padded_clean_manual_no_spk09_10 `
  --train-speakers spk03,spk04,spk05 `
  --val-speakers spk06 `
  --test-speakers spk07,spk08 `
  --exclude-speakers spk09,spk10 `
  --top-words 10
```

6. Rebuild `colab_lipreading_dataset.zip` from the cleaned split and cleaned mouth
crops.

## What To Say In The Coursework

After the first baseline experiment, visual inspection showed that automatic mouth
ROI extraction introduced noisy samples. A manual/semi-automatic filtering stage
was therefore added before training the final baseline model.
