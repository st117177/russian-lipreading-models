# Padded interval overlap audit

This is a read-only audit. It does not delete or rewrite clips.

Manifest: `C:\Users\Sobaka\Desktop\Lip-reading-demo-dataset\semester4_coursework\dataset_local\ru_dataset\01_intermediate_clips\word_clip_segments_bounded_v3.csv`
Total different-word overlap pairs: **11**
Selected for manual review: **11**

## Interpretation

Padding can create an overlap even when original word boundaries were correct. Review only checks whether neighbouring articulation is visible in the final crop.

## Overlap severity

- median overlap: `0.194` sec
- maximum overlap: `0.227` sec
- pairs with overlap >= 0.20 sec: `3`
- pairs with >= 50% of the shorter clip overlapped: `1`

## Pairs by speaker

- `spk16`: 3
- `spk17`: 1
- `spk19`: 7

## Review rule

Mark `keep` when the target word is isolated and `inspect` when a neighbour is visibly present. Only confirmed bad boundaries should be excluded later.
