#!/usr/bin/env python3
"""Batch-submit prepared MAUS wav/txt pairs and download TextGrid outputs."""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from submit_webmaus_basic import download_file, submit_job


RAW_VIDEOS_ROOT = Path("ru_dataset/raw_videos")


@dataclass
class BatchItem:
    video_dir: Path
    signal_path: Path
    text_path: Path
    output_path: Path


def find_text_files(maus_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in maus_dir.glob("*.txt")
        if path.is_file() and path.name != "transcript_raw.txt"
    )


def pick_signal_for_text(maus_dir: Path, text_path: Path) -> Path | None:
    exact = maus_dir / f"{text_path.stem}.wav"
    if exact.exists():
        return exact

    wavs = sorted(path for path in maus_dir.glob("*.wav") if path.is_file())
    if len(wavs) == 1:
        return wavs[0]
    return None


def iter_batch_items(raw_videos_root: Path) -> list[BatchItem]:
    items: list[BatchItem] = []

    for speaker_dir in sorted(path for path in raw_videos_root.iterdir() if path.is_dir()):
        if speaker_dir.name.startswith("."):
            continue
        for video_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            maus_dir = video_dir / "maus"
            align_dir = video_dir / "align"
            if not maus_dir.is_dir():
                continue

            for text_path in find_text_files(maus_dir):
                signal_path = pick_signal_for_text(maus_dir, text_path)
                if signal_path is None:
                    continue
                output_path = align_dir / f"{text_path.stem}.TextGrid"
                items.append(
                    BatchItem(
                        video_dir=video_dir,
                        signal_path=signal_path,
                        text_path=text_path,
                        output_path=output_path,
                    )
                )

    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit all prepared ru_dataset/raw_videos/*/*/maus wav/txt pairs to BAS WebMAUS"
    )
    parser.add_argument("--raw-videos-root", type=Path, default=RAW_VIDEOS_ROOT)
    parser.add_argument("--language", default="rus-RU")
    parser.add_argument("--out-format", default="TextGrid")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--speaker-id",
        action="append",
        default=None,
        help="Only process this speaker folder; repeat to select several speakers",
    )
    parser.add_argument("--video-id", default=None, help="Only process one video folder name")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    args = parser.parse_args()

    verify_ssl = not args.insecure
    items = iter_batch_items(args.raw_videos_root)

    if args.speaker_id:
        speaker_ids = set(args.speaker_id)
        items = [item for item in items if item.video_dir.parent.name in speaker_ids]
    if args.video_id is not None:
        items = [item for item in items if item.video_dir.name == args.video_id]
    if args.limit is not None and args.limit >= 0:
        items = items[: args.limit]

    if not items:
        print("No MAUS-ready wav/txt pairs found.")
        return 0

    done = 0
    skipped = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        if item.output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{len(items)}] SKIP existing: {item.output_path}")
            continue

        item.output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.backup_existing and item.output_path.exists():
            backup_path = item.output_path.with_suffix(item.output_path.suffix + ".bak")
            shutil.copy2(item.output_path, backup_path)
            print(f"[{idx}/{len(items)}] BACKUP {item.output_path} -> {backup_path}")

        print(
            f"[{idx}/{len(items)}] SUBMIT {item.video_dir.parent.name}/{item.video_dir.name} "
            f"-> {item.output_path.name}",
            flush=True,
        )

        try:
            download_link, warnings_text = submit_job(
                signal_path=item.signal_path,
                text_path=item.text_path,
                language=args.language,
                out_format=args.out_format,
                verify_ssl=verify_ssl,
                timeout_sec=args.timeout_sec,
            )
            download_file(
                download_link,
                item.output_path,
                verify_ssl=verify_ssl,
                timeout_sec=args.timeout_sec,
            )
            done += 1
            print(f"  saved: {item.output_path}")
            if warnings_text:
                print(f"  warnings: {warnings_text}")
        except Exception as exc:  # keep the batch moving
            failed += 1
            print(f"  ERROR: {exc!r}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print("")
    print(f"Processed: {len(items)}")
    print(f"Downloaded: {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
