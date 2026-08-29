#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DEFAULT_LABELS = Path("ru_dataset/selected_labels.csv")
DEFAULT_ROOT = Path("ru_dataset")
DEFAULT_OUT_DIR = Path("ru_dataset/quality")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sample_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    candidates = {0, frame_count // 2, max(0, frame_count - 1)}
    return sorted(candidates)


def detect_faces(frame: np.ndarray, detector: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def analyze_clip(path: Path, detector: cv2.CascadeClassifier) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "reason": "unreadable"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    brightness_values: list[float] = []
    blur_values: list[float] = []
    face_hits = 0

    for idx in sample_frame_indices(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(gray.mean()))
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if detect_faces(frame, detector):
            face_hits += 1

    cap.release()

    brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0.0
    blur = sum(blur_values) / len(blur_values) if blur_values else 0.0
    flags: list[str] = []
    if frame_count < 5:
        flags.append("too_few_frames")
    if duration < 0.18:
        flags.append("too_short")
    if duration > 2.5:
        flags.append("too_long")
    if brightness < 15:
        flags.append("very_dark")
    if brightness > 245:
        flags.append("very_bright")
    if face_hits == 0:
        flags.append("no_face_detected")

    return {
        "ok": True,
        "reason": "",
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "brightness": brightness,
        "blur": blur,
        "face_sample_hits": face_hits,
        "flags": ";".join(flags),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "word",
        "speaker_id",
        "clip_path",
        "ok",
        "reason",
        "fps",
        "frame_count",
        "width",
        "height",
        "duration_sec",
        "brightness",
        "blur",
        "face_sample_hits",
        "flags",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_clean_labels(
    labels_path: Path,
    quality_rows: list[dict[str, object]],
    output_path: Path,
    drop_flags: set[str],
) -> None:
    bad_clip_ids: set[str] = set()
    for row in quality_rows:
        flags = {flag for flag in str(row.get("flags") or "").split(";") if flag}
        if flags & drop_flags:
            bad_clip_ids.add(str(row["clip_id"]))

    with labels_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        clean_rows = [row for row in reader if row["clip_id"] not in bad_clip_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)


def make_contact_sheet(
    dataset_root: Path,
    quality_rows: list[dict[str, object]],
    out_path: Path,
    max_items: int,
    thumb_w: int = 180,
    thumb_h: int = 120,
) -> None:
    flagged = [row for row in quality_rows if row.get("flags")]
    if not flagged:
        flagged = quality_rows[:max_items]
    else:
        flagged = flagged[:max_items]

    thumbs: list[np.ndarray] = []
    for row in flagged:
        cap = cv2.VideoCapture(str(dataset_root / str(row["clip_path"])))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            frame = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        label = f"{row['clip_id']} {row['speaker_id']}"
        cv2.putText(thumb, label[:22], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        flag_text = str(row.get("flags") or "")[:28]
        if flag_text:
            cv2.putText(thumb, flag_text, (4, thumb_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
        thumbs.append(thumb)

    if not thumbs:
        return
    cols = 5
    rows_count = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows_count * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y = (i // cols) * thumb_h
        x = (i % cols) * thumb_w
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    flags = Counter()
    words = Counter()
    speakers = Counter()
    for row in rows:
        words[str(row["word"])] += 1
        speakers[str(row["speaker_id"])] += 1
        for flag in str(row.get("flags") or "").split(";"):
            if flag:
                flags[flag] += 1

    ok_rows = [row for row in rows if row["ok"]]
    durations = [float(row["duration_sec"]) for row in ok_rows]
    face_ok = sum(1 for row in ok_rows if int(row["face_sample_hits"]) > 0)
    lines = [
        "# Clip Quality Report",
        "",
        f"clips: {len(rows)}",
        f"readable: {len(ok_rows)}",
        f"with_face_detected_on_samples: {face_ok}",
    ]
    if durations:
        lines.extend(
            [
                f"duration_min: {min(durations):.3f}",
                f"duration_max: {max(durations):.3f}",
                f"duration_avg: {sum(durations) / len(durations):.3f}",
            ]
        )
    lines.extend(["", "## Flags"])
    if flags:
        lines.extend(f"- {flag}: {count}" for flag, count in flags.most_common())
    else:
        lines.append("- none")
    lines.extend(["", "## Speakers"])
    lines.extend(f"- {speaker}: {count}" for speaker, count in sorted(speakers.items()))
    lines.extend(["", "## Words"])
    lines.extend(f"- {word}: {count}" for word, count in words.most_common())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automatic quality checks on selected clips.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--contact-sheet-max", type=int, default=40)
    parser.add_argument("--clean-labels-output", type=Path, default=None)
    parser.add_argument(
        "--drop-flags",
        default="no_face_detected,very_dark,very_bright,too_few_frames,too_short,too_long,missing",
        help="Comma-separated quality flags to exclude from --clean-labels-output.",
    )
    args = parser.parse_args()

    rows = read_rows(args.labels)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise SystemExit(f"Could not load face cascade: {cascade_path}")

    quality_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        clip_path = args.dataset_root / row["clip_path"]
        if not clip_path.exists():
            analysis = {"ok": False, "reason": "missing", "flags": "missing"}
        else:
            analysis = analyze_clip(clip_path, detector)
        quality_rows.append({**row, **analysis})
        if idx % 100 == 0:
            print(f"checked {idx}/{len(rows)}")

    write_csv(args.out_dir / "clip_quality.csv", quality_rows)
    write_report(args.out_dir / "quality_report.md", quality_rows)
    make_contact_sheet(
        args.dataset_root,
        quality_rows,
        args.out_dir / "flagged_contact_sheet.jpg",
        args.contact_sheet_max,
    )
    if args.clean_labels_output is not None:
        drop_flags = {flag.strip() for flag in args.drop_flags.split(",") if flag.strip()}
        write_clean_labels(args.labels, quality_rows, args.clean_labels_output, drop_flags)
        print(f"Saved clean labels to {args.clean_labels_output}")
    print(f"Saved quality outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
