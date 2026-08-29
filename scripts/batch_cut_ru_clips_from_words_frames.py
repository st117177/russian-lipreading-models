#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cut_ru_clips_from_words_frames import build_segments, cut_segment, load_words_frames, sanitize_path_part


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS from {video_path}")
    return fps


def get_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Could not read duration from {video_path}")
    return frame_count / fps


def pick_video(chunk_dir: Path) -> Path | None:
    videos = sorted(chunk_dir.glob("*.mp4"))
    return videos[0] if len(videos) == 1 else None


def load_vocab(path: Path) -> set[str]:
    return {
        line.strip().lower().replace("ё", "е")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut selected word clips from aligned video chunks.")
    parser.add_argument("--chunks-root", type=Path, default=Path("ru_dataset/raw_video_chunks"))
    parser.add_argument("--clips-root", type=Path, default=Path("ru_dataset/selected_clips"))
    parser.add_argument("--vocab", type=Path, default=Path("ru_dataset/selected_vocab.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("ru_dataset/selected_clip_segments.csv"))
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--padding-frames", type=int, default=3)
    parser.add_argument("--start-padding-sec", type=float, default=0.20)
    parser.add_argument("--end-padding-sec", type=float, default=0.30)
    parser.add_argument("--max-duration-sec", type=float, default=2.0)
    parser.add_argument("--copy-codecs", action="store_true")
    parser.add_argument(
        "--speaker-id",
        action="append",
        default=None,
        help="Only process this speaker folder; repeat to select several speakers",
    )
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    speaker_ids = set(args.speaker_id or [])
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    rows: list[dict[str, str]] = []
    made = 0
    skipped = 0

    for wf in sorted(args.chunks_root.rglob("words_frames.txt")):
        align_dir = wf.parent
        chunk_dir = align_dir.parent
        speaker_id = chunk_dir.parent.name
        if speaker_ids and speaker_id not in speaker_ids:
            continue
        video_path = pick_video(chunk_dir)
        if video_path is None:
            skipped += 1
            continue
        fps = get_fps(video_path)
        source_duration = get_duration_sec(video_path)
        labels = [label.lower().replace("ё", "е") for label in load_words_frames(wf)]
        for idx, segment in enumerate(build_segments(labels, min_frames=args.min_frames), start=1):
            word = segment.word.lower().replace("ё", "е")
            if word not in vocab:
                continue

            start_sec = max(
                0.0,
                ((segment.start_frame - args.padding_frames) / fps) - args.start_padding_sec,
            )
            end_sec = min(
                source_duration,
                ((segment.end_frame + 1 + args.padding_frames) / fps) + args.end_padding_sec,
            )
            if end_sec <= start_sec or end_sec - start_sec > args.max_duration_sec:
                skipped += 1
                continue

            word_dir = sanitize_path_part(word)
            file_name = f"{chunk_dir.name}_{idx:06d}.mp4"
            output_path = args.clips_root / word_dir / speaker_id / file_name
            if output_path.exists():
                made += 1
            else:
                cut_segment(
                    ffmpeg_exe,
                    video_path,
                    output_path,
                    start_sec,
                    end_sec,
                    args.copy_codecs,
                    "veryfast",
                    False,
                )
                made += 1

            rows.append(
                {
                    "clip_file": output_path.as_posix(),
                    "word": word,
                    "speaker_id": speaker_id,
                    "start_frame": str(segment.start_frame),
                    "end_frame": str(segment.end_frame),
                    "start_sec": f"{start_sec:.3f}",
                    "end_sec": f"{end_sec:.3f}",
                    "source_video": video_path.as_posix(),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_file",
                "word",
                "speaker_id",
                "start_frame",
                "end_frame",
                "start_sec",
                "end_sec",
                "source_video",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Made/existing: {made}")
    print(f"Skipped: {skipped}")
    print(f"Manifest: {args.manifest}")
    print(f"Clips root: {args.clips_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
