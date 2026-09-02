# Scripts Map

This folder contains scripts from the original 3rd semester dataset pipeline and
scripts added or edited for the 4th semester extension.

Do not rename the `.py` files just to mark their status: the names are used in
commands, notes, and notebooks. The status is tracked here instead.

## Status Legend

- `3rd semester` - original script kept mostly as coursework foundation.
- `edited for 4th semester` - original script extended for the larger automated dataset.
- `new for 4th semester` - added for the expanded dataset, quality control, splits, or training preparation.

## Pipeline Order

```text
YouTube links
-> download videos/subtitles
-> split long videos into chunks
-> prepare wav + cleaned transcript for WebMAUS
-> WebMAUS forced alignment
-> TextGrid to words_frames.txt
-> cut word-level clips
-> build labels
-> quality check
-> create mouth crops
-> optionally create landmark-aligned mouth crops for model comparison
-> create train/val/test splits
-> package only split-referenced clips for Colab/Kaggle
-> train PyTorch baseline in Colab
```

## Script Table

| Script | Status | What it does | Main functions/classes |
| --- | --- | --- | --- |
| `download_candidate_videos.py` | new for 4th semester | Downloads YouTube videos and subtitles from a CSV list; `--max-height` selects the required source resolution. | `video_id_from_url`, `run_download`, `main` |
| `chunk_raw_videos_for_maus.py` | new for 4th semester | Splits long downloaded videos into shorter chunks for WebMAUS; `--max-duration-sec` can limit processing to the beginning of each source. | `pick_video`, `pick_srt`, `duration_sec`, `cut_video`, `iter_video_dirs` |
| `prepare_maus_inputs.py` | edited for 4th semester | Extracts `.wav` audio and prepares cleaned transcript text. Also handles SRT cleanup and uses the system `ffmpeg` with an `imageio-ffmpeg` fallback. | `parse_srt`, `clean_transcript`, `extract_audio`, `collapse_rolling_subtitle_text` |
| `batch_prepare_maus_inputs.py` | 3rd semester | Batch wrapper for preparing WebMAUS inputs. | `main` |
| `submit_webmaus_basic.py` | edited for 4th semester | Sends one audio/text pair to WebMAUS and downloads TextGrid; supports a bounded request timeout. | `submit_job`, `download_file`, `safe_upload_name` |
| `batch_submit_webmaus_basic.py` | edited for 4th semester | Batch wrapper for WebMAUS submission with speaker filtering, timeout handling, and per-file error isolation. | `iter_batch_items`, `find_text_files`, `pick_signal_for_text` |
| `get_phonewords_frames.py` | 3rd semester | Converts TextGrid word intervals to frame-level labels. | `make_words_frames_file`, `make_phonemes_frames_file`, `map_unknown_label` |
| `batch_generate_phonewords_frames.py` | edited for 4th semester | Batch wrapper for TextGrid to `words_frames.txt`; resolves the original script and phoneme dictionary relative to the repository. | `FrameJob`, `iter_jobs`, `run_job`, `get_fps`, `pick_textgrid` |
| `cut_ru_clips_from_words_frames.py` | 3rd semester | Cuts word-level clips from video using `words_frames.txt`. | `Segment`, `load_words_frames`, `build_segments`, `cut_segment`, `write_manifest` |
| `batch_cut_ru_clips_from_words_frames.py` | edited for 4th semester | Batch wrapper for cutting many word clips. Adds padded timing, repeatable speaker filtering, and OpenCV duration detection. | `get_fps`, `get_duration_sec`, `load_vocab`, `pick_video` |
| `build_labels_from_clips.py` | new for 4th semester | Builds CSV labels from the generated clip folder tree. | `collect_rows`, `get_clip_duration_sec`, `detect_source`, `write_labels` |
| `quality_check_clips.py` | new for 4th semester | Checks clip duration/readability/face detection, writes clean labels, and creates contact sheets labeled by `clip_id`. | `analyze_clip`, `detect_faces`, `write_clean_labels`, `make_contact_sheet`, `write_report` |
| `create_mouth_crops.py` | new for 4th semester | Creates mouth-region videos from word-level clips. | `find_face_box`, `mouth_box_from_face`, `fallback_lower_center_box`, `crop_video`, `write_manifest` |
| `create_landmark_mouth_crops.py` | new for 4th semester | Creates a separate experimental mouth ROI using per-frame MediaPipe landmarks, eye-line alignment, temporal smoothing, and detection diagnostics; supports the reserved final-test clip tree. | `detect_geometry`, `smooth_geometry`, `rotate_and_crop`, `crop_video`, `create_contact_sheet` |
| `apply_manual_mouth_crop_filter.py` | new for 4th semester | Removes manually rejected bad mouth crops from the labels CSV without deleting video files. | `normalize_path`, `load_bad_items`, `should_remove` |
| `create_ml_splits.py` | new for 4th semester | Creates random or speaker-based train/validation/test splits; `--only-speaker-top` avoids unused diagnostic split folders. | `top_words`, `filter_words`, `speaker_split`, `stratified_random_split`, `save_split_set` |
| `package_ml_dataset.py` | new for 4th semester | Builds a portable ZIP with only split-referenced clips and POSIX archive paths for Colab/Kaggle; can omit test while tuning. | `read_split_rows`, `archive_name`, `main` |
| `validate_ru_dataset.py` | 3rd semester | Validates dataset labels, vocabulary, and clip files. | `read_vocab`, `validate` |

## What Was Actually Added In Semester 4

The important new work is not "ten random scripts". It is four concrete blocks:

1. Automatic collection of more YouTube data.
2. Batch processing for long videos and WebMAUS.
3. Dataset quality control: padded clips, quality reports, mouth crops.
4. ML preparation: speaker-based splits, portable archive, and PyTorch baseline notebooks.

## Prior Work Attribution

The semester-3 `get_phonewords_frames.py` foundation and the compatible
`dicts/phonemes_keys.txt` mapping are based on the public
[Ustelemov/LipReading-RussianLang](https://github.com/Ustelemov/LipReading-RussianLang)
project used as inspiration for the original coursework. Semester-4 batch wrappers
call the local script directly; they do not require a second external copy.
