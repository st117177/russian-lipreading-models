#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

from prepare_maus_inputs import collapse_rolling_subtitle_text, parse_srt


def pick_video(video_dir: Path) -> Path | None:
    videos = sorted(video_dir.glob("*.mp4"))
    return videos[0] if len(videos) == 1 else None


def pick_srt(video_dir: Path) -> Path | None:
    video_id = video_dir.name
    for name in (f"{video_id}.ru.srt", f"{video_id}.ru-orig.srt"):
        candidate = video_dir / name
        if candidate.exists():
            return candidate
    srts = sorted(video_dir.glob("*.srt"))
    return srts[0] if srts else None


def duration_sec(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    return float(result.stdout.strip())


def cut_video(src: Path, dst: Path, start: float, length: float, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def iter_video_dirs(raw_root: Path) -> list[Path]:
    return [
        video_dir
        for speaker_dir in sorted(path for path in raw_root.iterdir() if path.is_dir())
        for video_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Split raw YouTube videos into MAUS-friendly chunks.")
    parser.add_argument("--raw-videos-root", type=Path, default=Path("ru_dataset/raw_videos"))
    parser.add_argument("--out-root", type=Path, default=Path("ru_dataset/raw_video_chunks"))
    parser.add_argument("--chunk-sec", type=float, default=300.0)
    parser.add_argument("--min-text-tokens", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--speaker-id",
        action="append",
        help="Only process this speaker. Repeat the option for multiple speakers.",
    )
    args = parser.parse_args()

    made = 0
    skipped = 0
    speaker_ids = set(args.speaker_id or [])
    for video_dir in iter_video_dirs(args.raw_videos_root):
        if speaker_ids and video_dir.parent.name not in speaker_ids:
            continue
        video_path = pick_video(video_dir)
        srt_path = pick_srt(video_dir)
        if video_path is None or srt_path is None:
            skipped += 1
            print(f"SKIP missing video/srt: {video_dir}")
            continue

        cues = parse_srt(srt_path)
        total = duration_sec(video_path)
        speaker_id = video_dir.parent.name
        video_id = video_dir.name
        chunks = int(math.ceil(total / args.chunk_sec))

        for idx in range(chunks):
            start = idx * args.chunk_sec
            end = min(total, (idx + 1) * args.chunk_sec)
            selected = [cue for cue in cues if start <= (cue.start + cue.end) / 2 < end]
            transcript = collapse_rolling_subtitle_text(selected)
            tokens = transcript.split()
            if len(tokens) < args.min_text_tokens:
                continue

            chunk_id = f"{video_id}_part{idx + 1:03d}"
            out_dir = args.out_root / speaker_id / chunk_id
            out_video = out_dir / f"{chunk_id}.mp4"
            out_txt = out_dir / f"{chunk_id}.sub.txt"

            cut_video(video_path, out_video, start, end - start, args.overwrite)
            if args.overwrite or not out_txt.exists():
                out_txt.write_text(transcript + "\n", encoding="utf-8")
            made += 1
            print(f"CHUNK {speaker_id}/{chunk_id}: {end - start:.1f}s, {len(tokens)} tokens")

    print(f"\nChunks: {made}")
    print(f"Skipped videos: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
