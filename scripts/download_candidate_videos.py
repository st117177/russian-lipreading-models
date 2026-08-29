#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


DEFAULT_MANIFEST = Path("candidate_videos.csv")
DEFAULT_RAW_ROOT = Path("ru_dataset/raw_videos")


def video_id_from_url(url: str) -> str:
    probe = subprocess.run(
        ["yt-dlp", "--ignore-config", "--print", "%(id)s", "--skip-download", "--no-warnings", url],
        text=True,
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    return probe.stdout.strip().splitlines()[-1]


def run_download(
    speaker_id: str, url: str, raw_root: Path, download_section: str = ""
) -> None:
    video_id = video_id_from_url(url)
    out_dir = raw_root / speaker_id / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "--ignore-config",
        "--no-playlist",
        "--no-write-thumbnail",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        "ru,ru-orig",
        "--convert-subs",
        "srt",
        "-f",
        "bv*[height<=480]+ba/b[height<=480]/b",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
    ]

    if download_section:
        cmd.extend(
            ["--download-sections", download_section, "--force-keyframes-at-cuts"]
        )

    cmd.append(url)
    print(f"\n== {speaker_id}/{video_id} ==")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download videos and Russian subtitles from a private CSV manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    failed = 0
    for row in rows:
        try:
            run_download(
                row["speaker_id"].strip(),
                row["url"].strip(),
                args.raw_root,
                row.get("download_section", "").strip(),
            )
        except Exception as exc:
            failed += 1
            print(f"ERROR {row['speaker_id']} {row['url']}: {exc!r}")

    print(f"\nFailed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
