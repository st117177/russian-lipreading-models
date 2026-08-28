# Dataset V1 Manifest

Dataset v1 is the frozen baseline dataset for the first controlled semester 4
experiments. Generated video files remain local and are not committed to GitHub.

## Baseline Split

```text
955 mouth-crop clips
10 word classes
6 speakers
speaker-based train/validation/test split
```

| Split | Clips | Speakers |
| --- | ---: | --- |
| train | 700 | spk03, spk04, spk05 |
| validation | 105 | spk06 |
| test | 150 | spk07, spk08 |

`spk09` and `spk10` are excluded because their generated clips/crops were not
reliable enough for the baseline.

Local split metadata:

```text
03_splits/ml_splits_mouth_crops_padded_clean_manual_no_spk09_10/speaker_top10/
```

Local video root:

```text
02_model_inputs/mouth_crops_padded/
```

## Counts by Word

| Word | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| будет | 80 | 5 | 18 | 103 |
| время | 62 | 4 | 7 | 73 |
| есть | 141 | 32 | 25 | 198 |
| значит | 82 | 4 | 6 | 92 |
| когда | 62 | 13 | 20 | 95 |
| может | 73 | 11 | 10 | 94 |
| потом | 58 | 9 | 3 | 70 |
| просто | 40 | 13 | 25 | 78 |
| человек | 43 | 5 | 8 | 56 |
| чтобы | 59 | 9 | 28 | 96 |

## Counts by Speaker

| Speaker | Clips |
| --- | ---: |
| spk03 | 123 |
| spk04 | 478 |
| spk05 | 99 |
| spk06 | 105 |
| spk07 | 60 |
| spk08 | 90 |

`spk04` accounts for about half of dataset v1. Dataset v2 should therefore add
new speakers instead of adding more `spk04` material. Priority classes for new
data are `человек`, `потом`, `время`, and `просто`.

## Evaluation Rules

- Use this speaker-based split for the main result.
- Use random split only as a diagnostic sanity check.
- Select a model using validation macro-F1, not test accuracy.
- Evaluate the final selected model on test once.
- Keep v1 unchanged so that future dataset v2 results remain comparable.
