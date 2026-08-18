#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path("ru_dataset")
DEFAULT_LABELS = Path("ru_dataset/selected_labels.csv")
DEFAULT_OUT_DIR = Path("ru_dataset/mouth_crops")
DEFAULT_MANIFEST = Path("ru_dataset/mouth_crops_labels.csv")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_crop_bounds(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    return x1, y1, x2, y2


def find_face_box(video_path: Path, detector: cv2.CascadeClassifier) -> tuple[int, int, int, int] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sorted({0, frame_count // 3, frame_count // 2, (2 * frame_count) // 3, max(0, frame_count - 1)})
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        for x, y, w, h in faces:
            area = int(w * h)
            if area > best_area:
                best = (int(x), int(y), int(w), int(h))
                best_area = area
    cap.release()
    return best


def mouth_box_from_face(
    face: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = face
    # Heuristic mouth ROI: centered lower face, expanded a little horizontally.
    mx = int(x + 0.12 * w)
    my = int(y + 0.55 * h)
    mw = int(0.76 * w)
    mh = int(0.38 * h)
    return safe_crop_bounds(mx, my, mw, mh, frame_width, frame_height)


def fallback_lower_center_box(frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    size = min(frame_width, frame_height)
    crop_w = int(size * 0.55)
    crop_h = int(size * 0.35)
    x = (frame_width - crop_w) // 2
    y = int(frame_height * 0.52)
    return safe_crop_bounds(x, y, crop_w, crop_h, frame_width, frame_height)


def crop_video(
    src: Path,
    dst: Path,
    detector: cv2.CascadeClassifier,
    size: int,
    use_fallback: bool,
) -> tuple[bool, str]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False, "unreadable"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    face = find_face_box(src, detector)
    if face is not None:
        box = mouth_box_from_face(face, width, height)
        status = "face_mouth_crop"
    elif use_fallback:
        box = fallback_lower_center_box(width, height)
        status = "fallback_lower_center_crop"
    else:
        cap.release()
        return False, "no_face"

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        cap.release()
        return False, "bad_crop_box"

    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps if fps > 0 else 25.0, (size, size))
    if not writer.isOpened():
        cap.release()
        return False, "writer_failed"

    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        written += 1
    writer.release()
    cap.release()

    if written == 0:
        return False, "no_frames_written"
    return True, status


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_dir(dataset_root: Path, out_dir: Path) -> tuple[Path, Path]:
    dataset_root = dataset_root.resolve()
    if out_dir.is_absolute():
        absolute_out_dir = out_dir.resolve()
    else:
        candidate = out_dir.resolve()
        if str(candidate).startswith(str(dataset_root)):
            absolute_out_dir = candidate
        else:
            absolute_out_dir = (dataset_root / out_dir).resolve()

    try:
        relative_out_dir = absolute_out_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise SystemExit(f"--out-dir must be inside --dataset-root: {absolute_out_dir}") from exc
    return absolute_out_dir, relative_out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Create heuristic mouth crops from selected clips.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fallback", action="store_true", help="Use lower-center crop when face is not detected.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.labels)
    if args.limit is not None:
        rows = rows[: args.limit]

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise SystemExit(f"Could not load face cascade: {cascade_path}")

    out_dir, rel_out_dir = resolve_output_dir(args.dataset_root, args.out_dir)
    out_rows: list[dict[str, str]] = []
    done = skipped = failed = 0
    for idx, row in enumerate(rows, start=1):
        src = args.dataset_root / row["clip_path"]
        rel_dst = rel_out_dir / row["word"] / row["speaker_id"] / Path(row["clip_path"]).name
        dst = out_dir / row["word"] / row["speaker_id"] / Path(row["clip_path"]).name
        if dst.exists() and not args.overwrite:
            ok, status = True, "existing"
            skipped += 1
        else:
            ok, status = crop_video(src, dst, detector, args.size, args.fallback)
        if ok:
            out_row = dict(row)
            out_row["clip_path"] = rel_dst.as_posix()
            out_row["crop_status"] = status
            out_rows.append(out_row)
            done += 1
        else:
            failed += 1
        if idx % 100 == 0:
            print(f"processed {idx}/{len(rows)} done={done} failed={failed}")

    fieldnames = list(rows[0].keys()) + ["crop_status"] if rows else []
    write_manifest(args.manifest, out_rows, fieldnames)
    print(f"done={done} skipped={skipped} failed={failed}")
    print(f"manifest={args.manifest}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
