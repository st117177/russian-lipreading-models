#!/usr/bin/env python3
"""
Cut Russian word clips from a source video using words_frames.txt.
ЛОГИКА РАБОТЫ
1. читаем words_frames.txt
2. находим последовательности одинаковых слов
3. переводим кадры → секунды
4. ffmpeg вырезает кусок видео
5. кладём файл в папку слова
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg

FORBIDDEN_CHARS = '<>:"/\\|?*'
SILENCE_LABELS = {
    "",
    "<silence:>",
    "<silence>",
    "<pause>",
    "<p:>",
    "sil",
    "sp",
}


@dataclass
class Segment:
    word: str
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1


def sanitize_path_part(value: str) -> str:
    """очищает слово, чтобы его можно было использовать как имя папки или файла"""
    cleaned = "".join("_" if ch in FORBIDDEN_CHARS else ch for ch in value.strip())
    cleaned = cleaned.strip(". ")
    return cleaned or "_"


def is_silence(label: str) -> bool:
    """проверяет это слово или пауза"""
    normalized = label.strip().lower()
    if normalized in SILENCE_LABELS:
        return True
    # для других тегов типо <laugh> тоже вернет True
    return normalized.startswith("<") and normalized.endswith(">")


def load_words_frames(path: Path) -> list[str]:
    """загружает файл words_frames.txt превращает его в список слов
    Послее этой функции у нас есть
    labels = [
    "привет",
    "привет",
    "привет",
    "как",
    "как",
    "дела"
    ]
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return [line.strip() for line in path.read_text(encoding=encoding).splitlines()]
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")
    # .splitlines() удаляет символы переноса :)
    # strip() удаляет пробелы


def build_segments(labels: list[str], min_frames: int) -> list[Segment]:
    """
    Превращает
    привет
    привет
    привет
    как
    как
    дела
    в
    Segment("привет",0,2)
    Segment("как",3,4)
    Segment("дела",5,5)
    """
    # создаётся пустой список, куда будут складываться найденные слова
    segments: list[Segment] = []  # type hint
    if not labels:
        return segments

    current_label = labels[0]
    start_frame = 0

    for frame_idx in range(1, len(labels) + 1):
        at_end = frame_idx == len(labels)
        next_label = None if at_end else labels[frame_idx]
        if at_end or next_label != current_label:
            # если текущее слово закончилось
            if current_label and not is_silence(current_label):
                segment = Segment(current_label, start_frame, frame_idx - 1)
                if segment.frame_count >= min_frames:
                    segments.append(segment)
            if not at_end:
                current_label = next_label
                start_frame = frame_idx

    return segments


def cut_segment(
    ffmpeg_exe: str,
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    copy_codecs: bool, # включён ли быстрый режим
    preset: str,
    low_memory_x264: bool,
) -> None:
    """режет видео"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # parents=True может создать недостающие родительские папки
    cmd = [
        ffmpeg_exe,
        "-y", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i",
        str(video_path),
        # делает команду ffmpeg -y -ss 12.300 -to 13.700 -i video.mp4
        "-map", "0:v:0",  # какие потоки из входного файла использовать
        "-map", "0:a:0?",  # ? - если аудио нет — не падай
    ]

    if copy_codecs:
        cmd.extend([
            "-c", "copy",  # быстро, но границы могут быть менее точными
        ])
        if low_memory_x264:
            # Reduce x264 buffering when the machine is low on virtual memory.
            cmd.extend([
                "-tune", "zerolatency",
                "-x264-params", "rc-lookahead=0:sync-lookahead=0:ref=1:bframes=0:scenecut=0",
            ])
    else:
        cmd.extend([
            "-c:v", "libx264",  # кодек H264
            "-preset", preset,  # можно ускорять кодирование без copy mode
            "-crf", "18",  # качество
            "-c:a", "aac",  # AAC — стандартный аудиокодек
            "-threads", "1",  # не даём x264 разгоняться в 22 потока и падать по памяти
        ])

    cmd.extend([
        "-movflags",
        "+faststart",
        str(output_path),
    ])

    subprocess.run(cmd, check=True)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """создаёт CSV-файл, где будет список всех вырезанных клипов"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut word clips from words_frames.txt")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--words-frames", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, default=Path("ru_dataset/clips"))
    parser.add_argument("--speaker-id", default="spk01")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--padding-frames", type=int, default=0)
    parser.add_argument("--start-padding-sec", type=float, default=0.0)
    parser.add_argument("--end-padding-sec", type=float, default=0.0)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--max-duration-sec", type=float, default=None)
    parser.add_argument("--max-source-seconds", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--clip-prefix", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--copy-codecs", action="store_true")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--low-memory-x264", action="store_true")
    args = parser.parse_args()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    labels = load_words_frames(args.words_frames)
    # -> получили список labels (подписей на каждый кадр)
    segments = build_segments(labels, min_frames=args.min_frames)
    # -> превратили в норм сегменты Segment("привет",0,2)

    clip_prefix = args.clip_prefix or f"{args.video.stem}_"
    # имя клипов
    manifest_path = args.manifest or args.words_frames.parent / "clip_segments.csv"
    # Путь к CSV

    rows: list[dict[str, str]] = []  # type hint
    skipped_too_long = 0
    skipped_past_source_cap = 0
    made = 0
    for idx, segment in enumerate(segments, start=1):
        if args.limit is not None and made >= args.limit:
            break
        word_dir = sanitize_path_part(segment.word)
        # папка с названием слова
        file_name = f"{clip_prefix}{idx:06d}.mp4"
        output_path = args.clips_root / word_dir / args.speaker_id / file_name
        # ну и сам путь
        start_sec = max(
            0.0,
            ((segment.start_frame - args.padding_frames) / args.fps) - args.start_padding_sec,
        )
        end_sec = (
            ((segment.end_frame + 1 + args.padding_frames) / args.fps) + args.end_padding_sec
        )

        if args.max_source_seconds is not None and end_sec > args.max_source_seconds:
            skipped_past_source_cap += 1
            continue

        duration_sec = end_sec - start_sec
        if args.max_duration_sec is not None and duration_sec > args.max_duration_sec:
            skipped_too_long += 1
            continue

        cut_segment(
            ffmpeg_exe,
            args.video,
            output_path,
            start_sec,
            end_sec,
            args.copy_codecs,
            args.preset,
            args.low_memory_x264,
        )
        rows.append(
            {
                "clip_file": output_path.as_posix(),
                "word": segment.word,
                "speaker_id": args.speaker_id,
                "start_frame": str(segment.start_frame),
                "end_frame": str(segment.end_frame),
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "source_video": args.video.as_posix(),
            }
        )
        made += 1

    write_manifest(manifest_path, rows)

    print(f"Segments: {len(segments)}")
    print(f"Manifest: {manifest_path}")
    print(f"Clips root: {args.clips_root}")
    print(f"Copy codecs: {args.copy_codecs}")
    print(f"Preset: {args.preset}")
    print(f"Skipped too long: {skipped_too_long}")
    print(f"Skipped past source cap: {skipped_past_source_cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
TextGrid
↓
words_frames.txt
↓
load_words_frames()
↓
labels = ["привет","привет","как"]
↓
build_segments(labels)
↓
Segment("привет",0,2)
↓
cut_segment
↓
ru_dataset/
   clips/
      слово/
         spk01/
            000001.mp4
            
+writes csv
"""
