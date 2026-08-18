#!/usr/bin/env python3
"""Batch-prepare MAUS wav/txt inputs from raw_videos speaker/video folders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from prepare_maus_inputs import extract_audio, get_video_id, read_text, clean_transcript


RAW_VIDEOS_ROOT = Path("ru_dataset/raw_videos")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class PrepareItem:
    video_dir: Path
    video_path: Path
    transcript_path: Path
    maus_dir: Path


def list_direct_files(video_dir: Path, suffixes: set[str]) -> list[Path]:
    return sorted(
        path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    )


def pick_video(video_dir: Path) -> Path | None:
    videos = list_direct_files(video_dir, VIDEO_EXTS)
    if len(videos) == 1:
        return videos[0]
    return None


def pick_transcript(video_dir: Path, video_path: Path) -> Path | None:
    candidates = list_direct_files(video_dir, {".txt"})
    if not candidates:
        return None

    preferred_names = [
        f"{video_path.stem}.txt",
        f"{video_path.stem}.sub.txt",
        f"{video_path.stem}.subtitles.txt",
    ]
    for name in preferred_names:
        candidate = video_dir / name
        if candidate.exists():
            return candidate

    if len(candidates) == 1:
        return candidates[0]
    return None


def iter_prepare_items(raw_videos_root: Path) -> list[PrepareItem]:
    items: list[PrepareItem] = []
    for speaker_dir in sorted(path for path in raw_videos_root.iterdir() if path.is_dir()):
        if speaker_dir.name.startswith("."):
            continue
        for video_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            video_path = pick_video(video_dir)
            if video_path is None:
                continue
            transcript_path = pick_transcript(video_dir, video_path)
            if transcript_path is None:
                continue
            items.append(
                PrepareItem(
                    video_dir=video_dir,
                    video_path=video_path,
                    transcript_path=transcript_path,
                    maus_dir=video_dir / "maus",
                )
            )
    return items


def outputs_exist(item: PrepareItem) -> bool:
    video_id = get_video_id(item.video_path)
    return (
        (item.maus_dir / f"{video_id}.txt").exists()
        and (item.maus_dir / f"{video_id}.wav").exists()
        and (item.maus_dir / "transcript_raw.txt").exists()
    )


def prepare_item(item: PrepareItem) -> None:
    item.maus_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_text(item.transcript_path)
    cleaned = clean_transcript(raw_text)
    video_id = get_video_id(item.video_path)

    (item.maus_dir / "transcript_raw.txt").write_text(raw_text, encoding="utf-8")
    (item.maus_dir / f"{video_id}.txt").write_text(cleaned + "\n", encoding="utf-8")
    extract_audio(item.video_path, item.maus_dir / f"{video_id}.wav")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare missing ru_dataset/raw_videos/*/*/maus wav/txt pairs"
    )
    parser.add_argument("--raw-videos-root", type=Path, default=RAW_VIDEOS_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--speaker-id", default=None, help="Only process one speaker folder, e.g. spk03")
    parser.add_argument("--video-id", default=None, help="Only process one video folder name")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = iter_prepare_items(args.raw_videos_root)
    if args.speaker_id is not None:
        items = [item for item in items if item.video_dir.parent.name == args.speaker_id]
    if args.video_id is not None:
        items = [item for item in items if item.video_dir.name == args.video_id]
    if args.limit is not None and args.limit >= 0:
        items = items[: args.limit]

    if not items:
        print("No video/transcript pairs found.")
        return 0

    prepared = 0
    skipped = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        if outputs_exist(item) and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{len(items)}] SKIP existing: {item.video_dir}")
            continue

        print(f"[{idx}/{len(items)}] PREPARE {item.video_dir.parent.name}/{item.video_dir.name}")
        try:
            prepare_item(item)
            prepared += 1
            print(f"  saved: {item.maus_dir}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR: {exc}")

    print("")
    print(f"Processed: {len(items)}")
    print(f"Prepared: {prepared}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
