#!/usr/bin/env python3
"""Prepare MAUS inputs from a source video and transcript text."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\s*$"
)
SRT_TIMESTAMP_RE = re.compile(
    r"^(?P<s>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<e>\d{2}:\d{2}:\d{2},\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def parse_srt_time(value: str) -> float:
    hh, mm, rest = value.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


def normalize_token(value: str) -> str:
    value = value.lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я-]+", "", value)


def parse_srt(path: Path) -> list[Cue]:
    cues: list[Cue] = []
    start: float | None = None
    end: float | None = None
    text_lines: list[str] = []

    def flush() -> None:
        nonlocal start, end, text_lines
        if start is not None and end is not None and text_lines:
            text = TAG_RE.sub(" ", " ".join(text_lines))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                cues.append(Cue(start, end, text))
        start = None
        end = None
        text_lines = []

    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.isdigit():
            continue
        match = SRT_TIMESTAMP_RE.match(line)
        if match:
            flush()
            start = parse_srt_time(match.group("s"))
            end = parse_srt_time(match.group("e"))
            continue
        text_lines.append(line)
    flush()
    return cues


def collapse_rolling_subtitle_text(cues: list[Cue]) -> str:
    """Collapse repeated tokens common in YouTube auto-subtitle windows."""
    out_tokens: list[str] = []
    prev_tokens: list[str] = []
    for cue in cues:
        cue_tokens = [token for token in cue.text.split() if normalize_token(token)]
        if not cue_tokens:
            continue
        max_overlap = min(len(prev_tokens), len(cue_tokens))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            prev_norm = [normalize_token(token) for token in prev_tokens[-size:]]
            cue_norm = [normalize_token(token) for token in cue_tokens[:size]]
            if prev_norm == cue_norm:
                overlap = size
                break
        out_tokens.extend(cue_tokens[overlap:])
        prev_tokens = cue_tokens
    return " ".join(out_tokens)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".srt":
        return collapse_rolling_subtitle_text(parse_srt(path))
    return path.read_text(encoding="utf-8-sig")


def clean_transcript(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("WEBVTT"):
            continue
        if TIMESTAMP_RE.match(line):
            continue
        if line.isdigit():
            continue
        lines.append(line)

    merged = " ".join(lines)
    merged = re.sub(r"<[^>]+>", " ", merged)
    merged = re.sub(r"\[[^\]]+\]", " ", merged)
    merged = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\- ]+", " ", merged)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged.lower()


def extract_audio(video_path: Path, audio_path: Path) -> None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)


def get_video_id(video_path: Path) -> str:
    stem = video_path.stem
    if stem.startswith("yt_"):
        stem = stem[3:]
    if stem.endswith(".sub"):
        stem = stem[:-4]
    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare <video_id>.wav and <video_id>.txt for MAUS")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_text(args.transcript)
    cleaned = clean_transcript(raw_text)
    video_id = args.video_id or get_video_id(args.video)

    raw_out = args.out_dir / "transcript_raw.txt"
    maus_out = args.out_dir / f"{video_id}.txt"
    audio_out = args.out_dir / f"{video_id}.wav"

    raw_out.write_text(raw_text, encoding="utf-8")
    maus_out.write_text(cleaned + "\n", encoding="utf-8")
    extract_audio(args.video, audio_out)

    print(f"Saved raw transcript: {raw_out}")
    print(f"Saved MAUS transcript: {maus_out}")
    print(f"Saved audio: {audio_out}")
    print(f"MAUS transcript chars: {len(cleaned)}")
    print(f"MAUS transcript words: {len(cleaned.split())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
