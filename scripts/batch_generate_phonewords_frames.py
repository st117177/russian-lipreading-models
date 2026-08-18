#!/usr/bin/env python3
"""Batch-generate words_frames.txt and related frame files from TextGrid alignments.
1. ищет все папки

ru_dataset/raw_videos/spk*/video_id

2. находит

video.mp4
align/*.TextGrid

3. читает FPS

cv2.VideoCapture(video)

4. вызывает

external/ustelemov/scripts/get_phonewords_frames.py"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

from prepare_maus_inputs import get_video_id


RAW_VIDEOS_ROOT = Path("ru_dataset/raw_videos")
PHONEWORDS_SCRIPT = Path("external/ustelemov/scripts/get_phonewords_frames.py")
PHONEME_KEYS_DICT = Path("external/ustelemov/dicts/phonemes_keys.txt")
EXCLUDED_VIDEOS_PATH = Path("ru_dataset/qa/excluded_videos.csv")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
OUTPUT_FILES = (
    "words_frames.txt",
    "phonemes_frames.txt",
    "phonemekeys_frames.txt",
    "phonemeswords_frames.txt",
)

# контейнер с информацией про одно видео
@dataclass
class FrameJob: # задание по генерации frame-файлов
    video_dir: Path
    video_path: Path
    textgrid_path: Path
    align_dir: Path
    fps: float

# вспомогательные функции для поиска видео, текстгридов и получения FPS
def pick_video(video_dir: Path) -> Path | None:
    videos = sorted(
        path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    if len(videos) == 1:
        return videos[0]
    return None

def pick_textgrid(video_path: Path, align_dir: Path) -> Path | None:
    textgrids = sorted(path for path in align_dir.glob("*.TextGrid") if path.is_file())
    if not textgrids:
        return None

    video_id = get_video_id(video_path)
    preferred = [
        align_dir / f"{video_id}.TextGrid",
        align_dir / "maus_out.TextGrid",
        align_dir / "audio.TextGrid",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    if len(textgrids) == 1:
        return textgrids[0]
    return None


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()
    if fps <= 0:
        raise RuntimeError(f"Could not read FPS from {video_path}")
    return fps

def outputs_exist(align_dir: Path) -> bool:
    return all((align_dir / name).exists() for name in OUTPUT_FILES)

def load_excluded_videos(path: Path) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    if not path.exists():
        return excluded

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("status") or "").strip().lower() != "excluded":
                continue
            speaker_id = (row.get("speaker_id") or "").strip()
            video_dir = (row.get("video_dir") or "").strip()
            if speaker_id and video_dir:
                excluded.add((speaker_id, video_dir))
    return excluded

def iter_jobs(raw_videos_root: Path) -> list[FrameJob]: 
    """Итератор по всем видео с выравниванием, возвращающий FrameJob для каждого видео
    для каждой папки спикера, для каждого видео, если есть папка align,
    если есть видео, если есть текстгрид, если можно прочитать FPS, то создать FrameJob и добавить его 
    в список задач"""
    jobs: list[FrameJob] = [] # список для хранения всех найденных задач
    for speaker_dir in sorted(path for path in raw_videos_root.iterdir() if path.is_dir()):
        if speaker_dir.name.startswith("."): # скрытая папка
            continue
        for video_dir in sorted(path for path in speaker_dir.iterdir() if path.is_dir()):
            align_dir = video_dir / "align"
            if not align_dir.is_dir():
                continue
            video_path = pick_video(video_dir)
            if video_path is None:
                continue
            textgrid_path = pick_textgrid(video_path, align_dir)
            if textgrid_path is None:
                continue
            fps = get_fps(video_path)
            jobs.append(
                FrameJob(
                    video_dir=video_dir,
                    video_path=video_path,
                    textgrid_path=textgrid_path,
                    align_dir=align_dir,
                    fps=fps,
                )
            )
    return jobs


def run_job(job: FrameJob, phoneme_keys_dict: Path, script_path: Path) -> None:
    """Вызывает внешний скрипт get_phonewords_frames.py с аргументами"""
    cmd = [
        sys.executable,
        str(script_path),
        "--f",
        f"{job.fps:.6f}",
        "--p",
        str(job.align_dir),
        "--t",
        str(job.textgrid_path),
        "--d",
        str(phoneme_keys_dict),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate words_frames.txt and related files for all aligned raw videos"
    )
    parser.add_argument("--raw-videos-root", type=Path, default=RAW_VIDEOS_ROOT)
    parser.add_argument("--script", type=Path, default=PHONEWORDS_SCRIPT)
    parser.add_argument("--phoneme-keys-dict", type=Path, default=PHONEME_KEYS_DICT)
    parser.add_argument("--excluded-videos", type=Path, default=EXCLUDED_VIDEOS_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    # зачем это нужно? чтобы можно было запустить скрипт только для одного спикера или одного видео, или ограничить количеством видео для теста
    # так мы же весь этот код и написали чтобы сразу ВСЕ видео обработать
    # но иногда может быть полезно запустить на одном видео, чтобы проверить что все работает, 
    # или запустить на 10 видео, чтобы посмотреть статистику по ошибкам, прежде чем запускать на всех видео
    args = parser.parse_args()

    jobs = iter_jobs(args.raw_videos_root)
    excluded_videos = load_excluded_videos(args.excluded_videos)
    skipped_excluded = 0
    if excluded_videos:
        filtered_jobs: list[FrameJob] = []
        for job in jobs:
            key = (job.video_dir.parent.name, job.video_dir.name)
            if key in excluded_videos:
                skipped_excluded += 1
                continue
            filtered_jobs.append(job)
        jobs = filtered_jobs
    if args.speaker_id is not None:
        jobs = [job for job in jobs if job.video_dir.parent.name == args.speaker_id] 
        # фильтруем задачи по спикеру
    if args.video_id is not None:
        jobs = [job for job in jobs if job.video_dir.name == args.video_id]
        # фильтруем задачи по видео
    if args.limit is not None and args.limit >= 0:
        jobs = jobs[: args.limit] # если есть лимит

    if not jobs:
        print("No eligible aligned videos found.") # перевод "Не найдено подходящих выровненных видео."
        return 0

    generated = 0
    skipped = 0
    failed = 0
    # считаем сколько задач обработано, сколько пропущено из-за существующих выходных файлов, 
    # сколько завершилось с ошибкой
    for idx, job in enumerate(jobs, start=1):
        if outputs_exist(job.align_dir) and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{len(jobs)}] SKIP existing: {job.video_dir}")
            continue

        print(
            f"[{idx}/{len(jobs)}] GENERATE {job.video_dir.parent.name}/{job.video_dir.name} "
            f"using {job.textgrid_path.name} @ {job.fps:.3f} fps"
        )
        try:
            run_job(job, args.phoneme_keys_dict, args.script)
            generated += 1
            print(f"  saved: {job.align_dir}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR: {exc!r}")

    print("")
    print(f"Processed: {len(jobs)}")
    print(f"Generated: {generated}")
    print(f"Skipped: {skipped}")
    print(f"Excluded: {skipped_excluded}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
