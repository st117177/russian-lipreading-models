#!/usr/bin/env python3
"""Build labels.csv from a clip folder structure.

Expected clip layout:
<dataset_root>/<clips_dir>/<word>/<speaker_id>/<video_file>
"""

import argparse
import csv
from pathlib import Path

import imageio_ffmpeg

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
HEADER = [
    "clip_id",
    "clip_path",
    "word",
    "speaker_id",
    "source",
    "start_sec",
    "end_sec",
]


def get_clip_duration_sec(clip_file: Path) -> float:
    """Return clip duration in seconds for labels.csv metadata."""
    _, duration_sec = imageio_ffmpeg.count_frames_and_secs(str(clip_file))
    return max(float(duration_sec), 0.001)


def detect_source(file_name: str, default_source: str) -> str:
    name = file_name.lower()
    if name.startswith("yt_") or "youtube" in name:
        return "youtube"
    if name.startswith("self_"):
        return "self_recorded"
    return default_source


def collect_rows(
    dataset_root: Path,
    clips_dir: str,
    default_source: str,
    clip_prefix: str,
) -> list[list[str]]:
    clips_root = dataset_root / clips_dir
    rows: list[list[str]] = []

    if not clips_root.exists():
        return rows

    clip_index = 1
    for word_dir in sorted(p for p in clips_root.iterdir() if p.is_dir()):
        word = word_dir.name
        for speaker_dir in sorted(p for p in word_dir.iterdir() if p.is_dir()):
            speaker_id = speaker_dir.name
            files = sorted(
                p for p in speaker_dir.iterdir()
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
            )
            for clip_file in files:
                clip_id = f"{clip_prefix}{clip_index:06d}"
                clip_index += 1
                rel_path = clip_file.relative_to(dataset_root).as_posix()
                source = detect_source(clip_file.name, default_source)
                duration_sec = get_clip_duration_sec(clip_file)
                rows.append([
                    clip_id,
                    rel_path,
                    word,
                    speaker_id,
                    source,
                    "0.0",
                    f"{duration_sec:.3f}",
                ])

    return rows


def write_labels(labels_path: Path, rows: list[list[str]]) -> None:
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build labels.csv from a clip folder tree")
    parser.add_argument("--dataset-root", type=Path, default=Path("ru_dataset"))
    parser.add_argument("--clips-dir", default="clips")
    parser.add_argument("--output", type=Path, default=None, help="Default: <dataset-root>/labels.csv")
    parser.add_argument("--vocab-output", type=Path, default=None)
    parser.add_argument("--default-source", default="self_recorded", choices=["self_recorded", "youtube"])
    parser.add_argument("--clip-prefix", default="ru_")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output = args.output if args.output is not None else dataset_root / "labels.csv"

    rows = collect_rows(dataset_root, args.clips_dir, args.default_source, args.clip_prefix)
    write_labels(output, rows)

    if args.vocab_output is not None:
        words = sorted({row[2] for row in rows})
        args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
        args.vocab_output.write_text("\n".join(words) + "\n", encoding="utf-8")

    print(f"Built labels: {output}")
    print(f"Rows: {len(rows)}")
    if args.vocab_output is not None:
        print(f"Built vocab: {args.vocab_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
