#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import mediapipe as mp
import numpy as np


DEFAULT_DATASET_ROOT = Path("ru_dataset")
DEFAULT_OUT_DIR = Path("02_model_inputs/mouth_crops_landmark_v3")
DEFAULT_MANIFEST = Path("04_quality_reports/landmark_v3_labels.csv")

# Canonical MediaPipe Face Mesh lip contour indices.
LIP_INDICES = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
)
LEFT_EYE_INDICES = (33, 133)
RIGHT_EYE_INDICES = (362, 263)

SOURCE_PREFIX_MAP = {
    "02_model_inputs/mouth_crops_padded": "01_intermediate_clips/selected_clips_padded",
    "02_model_inputs/mouth_crops_padded_v2_new": "01_intermediate_clips/word_clips_padded_v2_new",
}


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                key = row.get("clip_id") or row["clip_path"]
                if key not in seen:
                    rows.append(row)
                    seen.add(key)
    return rows


def sample_balanced(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    if size >= len(rows):
        return rows
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["speaker_id"], row["word"])].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, str]] = []
    keys = list(groups)
    rng.shuffle(keys)
    while len(selected) < size:
        made_progress = False
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def map_crop_to_source(clip_path: str) -> PurePosixPath:
    normalized = PurePosixPath(clip_path.replace("\\", "/"))
    value = normalized.as_posix()
    for crop_prefix, source_prefix in SOURCE_PREFIX_MAP.items():
        if value == crop_prefix or value.startswith(crop_prefix + "/"):
            suffix = value[len(crop_prefix):].lstrip("/")
            return PurePosixPath(source_prefix) / suffix
    raise ValueError(f"No source mapping for clip path: {clip_path}")


def landmark_point(landmarks: list, index: int, width: int, height: int) -> np.ndarray:
    point = landmarks[index]
    return np.array([point.x * width, point.y * height], dtype=np.float32)


def detect_geometry(frame: np.ndarray, face_mesh) -> tuple[float, float, float, float] | None:
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    lips = np.stack([landmark_point(landmarks, index, width, height) for index in LIP_INDICES])
    left_eye = np.mean(
        [landmark_point(landmarks, index, width, height) for index in LEFT_EYE_INDICES], axis=0
    )
    right_eye = np.mean(
        [landmark_point(landmarks, index, width, height) for index in RIGHT_EYE_INDICES], axis=0
    )
    eye_a, eye_b = sorted((left_eye, right_eye), key=lambda point: point[0])
    eye_delta = eye_b - eye_a
    inter_eye = float(np.linalg.norm(eye_delta))
    if inter_eye < 8.0:
        return None

    center = lips.mean(axis=0)
    lip_width = float(np.ptp(lips[:, 0]))
    lip_height = float(np.ptp(lips[:, 1]))
    side = max(1.30 * inter_eye, 1.75 * lip_width, 3.00 * lip_height)
    angle = float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0])))
    return float(center[0]), float(center[1]), side, angle


def interpolate_missing(values: list[tuple[float, float, float, float] | None]) -> np.ndarray | None:
    valid_indices = np.array([index for index, value in enumerate(values) if value is not None])
    if valid_indices.size == 0:
        return None
    valid_values = np.array([values[index] for index in valid_indices], dtype=np.float32)
    all_indices = np.arange(len(values))
    columns = [np.interp(all_indices, valid_indices, valid_values[:, column]) for column in range(4)]
    return np.stack(columns, axis=1).astype(np.float32)


def smooth_geometry(values: np.ndarray, radius: int = 2) -> np.ndarray:
    smoothed = values.copy()
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        smoothed[index] = np.median(values[start:stop], axis=0)
    return smoothed


def rotate_and_crop(frame: np.ndarray, geometry: np.ndarray, output_size: int) -> np.ndarray:
    center_x, center_y, side, angle = map(float, geometry)
    height, width = frame.shape[:2]
    rotation = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    aligned = cv2.warpAffine(
        frame,
        rotation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    side_int = max(16, int(round(side)))
    crop = cv2.getRectSubPix(aligned, (side_int, side_int), (center_x, center_y))
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def crop_video(
    source: Path,
    destination: Path,
    output_size: int,
    min_detection_ratio: float,
) -> tuple[bool, dict[str, str]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return False, {"landmark_status": "unreadable_source"}

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    geometry: list[tuple[float, float, float, float] | None] = []
    # Tracking is useful within one word clip, but its state must not leak into
    # the next clip, which can contain a completely different speaker.
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50,
    ) as face_mesh:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            geometry.append(detect_geometry(frame, face_mesh))
    capture.release()

    if not frames:
        return False, {"landmark_status": "no_frames"}
    hits = sum(value is not None for value in geometry)
    detection_ratio = hits / len(frames)
    diagnostics = {
        "landmark_status": "ok",
        "landmark_frames": str(hits),
        "total_frames": str(len(frames)),
        "landmark_detection_ratio": f"{detection_ratio:.4f}",
    }
    if detection_ratio < min_detection_ratio:
        diagnostics["landmark_status"] = "low_detection_ratio"
        return False, diagnostics

    interpolated = interpolate_missing(geometry)
    if interpolated is None:
        diagnostics["landmark_status"] = "no_landmarks"
        return False, diagnostics
    smoothed = smooth_geometry(interpolated)

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25.0,
        (output_size, output_size),
    )
    if not writer.isOpened():
        diagnostics["landmark_status"] = "writer_failed"
        return False, diagnostics
    for frame, frame_geometry in zip(frames, smoothed):
        writer.write(rotate_and_crop(frame, frame_geometry, output_size))
    writer.release()

    diagnostics["interpolated_frames"] = str(len(frames) - hits)
    diagnostics["mean_crop_side_px"] = f"{float(smoothed[:, 2].mean()):.2f}"
    return True, diagnostics


def middle_frame(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def create_contact_sheet(pairs: list[tuple[Path, Path]], output: Path, tile_size: int = 150) -> None:
    tiles: list[np.ndarray] = []
    for old_path, new_path in pairs:
        old_frame = middle_frame(old_path)
        new_frame = middle_frame(new_path)
        if old_frame is None or new_frame is None:
            continue
        old_frame = cv2.resize(old_frame, (tile_size, tile_size))
        new_frame = cv2.resize(new_frame, (tile_size, tile_size))
        cv2.putText(old_frame, "old", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(new_frame, "landmark", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 0), 2)
        tiles.append(np.hstack([old_frame, new_frame]))
    if not tiles:
        return
    columns = 4
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create temporally smoothed MediaPipe mouth crops from train/validation split CSV files."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-csv", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--failed-manifest", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--min-detection-ratio", type=float, default=0.70)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else dataset_root / args.out_dir
    out_dir = out_dir.resolve()
    try:
        relative_out_dir = out_dir.relative_to(dataset_root)
    except ValueError as exc:
        raise SystemExit("--out-dir must be inside --dataset-root") from exc

    rows = read_rows(args.split_csv)
    if args.sample_size is not None:
        rows = sample_balanced(rows, args.sample_size, args.seed)

    output_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    contact_pairs: list[tuple[Path, Path]] = []
    for index, row in enumerate(rows, start=1):
        try:
            source_relative = map_crop_to_source(row["clip_path"])
        except ValueError as exc:
            failed = dict(row)
            failed["landmark_status"] = str(exc)
            failed_rows.append(failed)
            continue

        source = dataset_root / Path(source_relative.as_posix())
        destination_relative = (
            PurePosixPath(relative_out_dir.as_posix())
            / row["word"]
            / row["speaker_id"]
            / source.name
        )
        destination = dataset_root / Path(destination_relative.as_posix())
        if destination.exists() and not args.overwrite:
            ok = True
            diagnostics = {"landmark_status": "existing"}
        else:
            ok, diagnostics = crop_video(
                source,
                destination,
                args.size,
                args.min_detection_ratio,
            )

        output_row = dict(row)
        output_row.update(diagnostics)
        output_row["source_clip_path"] = source_relative.as_posix()
        if ok:
            old_crop = dataset_root / Path(PurePosixPath(row["clip_path"]).as_posix())
            output_row["clip_path"] = destination_relative.as_posix()
            output_rows.append(output_row)
            if len(contact_pairs) < 24:
                contact_pairs.append((old_crop, destination))
        else:
            failed_rows.append(output_row)

        if index % 25 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)} ok={len(output_rows)} failed={len(failed_rows)}")

    write_csv(args.manifest, output_rows)
    failed_manifest = args.failed_manifest or args.manifest.with_name(args.manifest.stem + "_failed.csv")
    write_csv(failed_manifest, failed_rows)
    if args.contact_sheet:
        create_contact_sheet(contact_pairs, args.contact_sheet)

    print(f"manifest={args.manifest}")
    print(f"failed_manifest={failed_manifest}")
    print(f"output_dir={out_dir}")
    return 0 if output_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
