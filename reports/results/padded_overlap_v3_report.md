# Padded interval overlap audit

This is a read-only audit. It does not delete or rewrite clips.

Manifest: `C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\01_intermediate_clips\word_clip_segments_padded_v3_new.csv`
Total different-word overlap pairs: **45**
Selected for manual review: **30**

## Interpretation

Padding can create an overlap even when original word boundaries were correct. Review only checks whether neighbouring articulation is visible in the final crop.

## Pairs by speaker

- `spk16`: 25
- `spk17`: 7
- `spk18`: 3
- `spk19`: 10

## Review rule

Mark `keep` when the target word is isolated and `inspect` when a neighbour is visibly present. Only confirmed bad boundaries should be excluded later.
