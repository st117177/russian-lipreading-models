# Padded interval overlap audit

This is a read-only audit. It does not delete or rewrite clips.

Manifest: `C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\01_intermediate_clips\final_test_clip_segments_padded.csv`
Total different-word overlap pairs: **26**
Selected for manual review: **26**

## Interpretation

Padding can create an overlap even when original word boundaries were correct. Review only checks whether neighbouring articulation is visible in the final crop.

## Pairs by speaker

- `spk13`: 16
- `spk14`: 8
- `spk15`: 2

## Review rule

Mark `keep` when the target word is isolated and `inspect` when a neighbour is visibly present. Only confirmed bad boundaries should be excluded later.
