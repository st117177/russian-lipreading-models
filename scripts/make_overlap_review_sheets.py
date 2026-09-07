#!/usr/bin/env python3
"""Create visual review sheets for the overlap audit CSV."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_font(path: str, size: int):
    font_path = Path(path)
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def extract_frames(
    ffmpeg: str,
    source: Path,
    duration: float,
    output_prefix: Path,
) -> list[Image.Image]:
    fps = 3.0 / max(duration, 0.12)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps:.6f},scale=128:128",
        "-frames:v",
        "3",
        "-q:v",
        "3",
        f"{output_prefix}_%02d.jpg",
    ]
    subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    images: list[Image.Image] = []
    for path in sorted(output_prefix.parent.glob(f"{output_prefix.name}_*.jpg"))[:3]:
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        except OSError:
            continue
    if not images:
        images.append(Image.new("RGB", (128, 128), "black"))
    while len(images) < 3:
        images.append(images[-1].copy())
    return images[:3]


def create_sheets(
    rows: list[dict[str, str]],
    ffmpeg: str,
    output_dir: Path,
    font_path: str,
    rows_per_sheet: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="overlap-review-"))
    try:
        title_font = load_font(font_path, 16)
        caption_font = load_font(font_path, 13)
        frame_width = 128
        row_height = 190
        header_height = 34

        prepared: list[tuple[dict[str, str], list[Image.Image]]] = []
        for index, row in enumerate(rows, start=1):
            pair_dir = temporary_dir / f"pair_{index:02d}"
            pair_dir.mkdir()
            frames: list[Image.Image] = []
            for side in ("a", "b"):
                source = Path(row[f"clip_file_{side}"].replace("/", "\\"))
                frames.extend(
                    extract_frames(
                        ffmpeg,
                        source,
                        float(row[f"duration_sec_{side}"]),
                        pair_dir / side,
                    )
                )
            prepared.append((row, frames))

        for page_index, start in enumerate(range(0, len(prepared), rows_per_sheet), start=1):
            page_rows = prepared[start : start + rows_per_sheet]
            sheet = Image.new(
                "RGB",
                (820, header_height + row_height * len(page_rows)),
                "white",
            )
            draw = ImageDraw.Draw(sheet)
            for local_index, (row, frames) in enumerate(page_rows):
                y = header_height + local_index * row_height
                title = (
                    f"pair {start + local_index + 1}: {row['word_a']} -> {row['word_b']} | "
                    f"overlap={float(row['overlap_sec']):.3f}s | "
                    f"ratio={float(row['overlap_ratio_shorter']):.2f}"
                )
                draw.text((8, y + 5), title, fill="black", font=title_font)
                for frame_index, frame in enumerate(frames):
                    x = 8 + frame_index * (frame_width + 4)
                    sheet.paste(frame, (x, y + 30))
                draw.text((8, y + 162), f"A: {row['clip_id_a']}", fill="black", font=caption_font)
                draw.text((410, y + 162), f"B: {row['clip_id_b']}", fill="black", font=caption_font)

            output = output_dir / f"moderate_overlap_review_sheet_{page_index}.jpg"
            sheet.save(output, quality=92)
            print(output)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create visual sheets for overlap review.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-exe", required=True)
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arial.ttf")
    parser.add_argument("--rows-per-sheet", type=int, default=7)
    args = parser.parse_args()

    rows = read_rows(args.review_csv)
    create_sheets(rows, args.ffmpeg_exe, args.output_dir, args.font, args.rows_per_sheet)
    print(f"pairs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
