#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def normalize_path(value: str) -> str:
    normalized = value.strip().strip('"').strip("'").replace("\\", "/")
    marker = "ru_dataset/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    model_inputs_prefix = "02_model_inputs/"
    if normalized.startswith(model_inputs_prefix):
        normalized = normalized[len(model_inputs_prefix) :]
    return normalized.lstrip("./")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_bad_items(path: Path) -> tuple[set[str], set[str], Counter[str]]:
    if not path.exists():
        raise SystemExit(f"Manual bad-crops file not found: {path}")

    rows = read_rows(path)
    bad_clip_paths: set[str] = set()
    bad_clip_ids: set[str] = set()
    reasons: Counter[str] = Counter()

    for row in rows:
        clip_path = normalize_path(row.get("clip_path", ""))
        clip_id = row.get("clip_id", "").strip()
        reason = row.get("reason", "").strip() or "unspecified"

        if clip_path:
            bad_clip_paths.add(clip_path)
            reasons[reason] += 1
        if clip_id:
            bad_clip_ids.add(clip_id)
            reasons[reason] += 1

    return bad_clip_paths, bad_clip_ids, reasons


def should_remove(row: dict[str, str], bad_paths: set[str], bad_ids: set[str]) -> bool:
    clip_path = normalize_path(row.get("clip_path", ""))
    clip_id = row.get("clip_id", "").strip()
    return clip_path in bad_paths or clip_id in bad_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove manually rejected mouth crops from a labels CSV."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("02_model_inputs/mouth_crops_padded_labels.csv"),
        help="Input labels CSV with mouth crop paths.",
    )
    parser.add_argument(
        "--bad-list",
        type=Path,
        default=Path("02_model_inputs/manual_bad_mouth_crops.csv"),
        help="CSV with bad clips. Columns: clip_path,reason,notes. clip_id is also supported.",
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=Path("02_model_inputs/mouth_crops_padded_labels_clean_manual.csv"),
        help="Output labels CSV after manual filtering.",
    )
    parser.add_argument(
        "--removed-out",
        type=Path,
        default=Path("02_model_inputs/manual_bad_mouth_crops_removed_rows.csv"),
        help="CSV with rows removed from the labels file.",
    )
    args = parser.parse_args()

    rows = read_rows(args.labels)
    if not rows:
        raise SystemExit(f"No rows found in labels file: {args.labels}")

    bad_paths, bad_ids, reasons = load_bad_items(args.bad_list)
    kept_rows = []
    removed_rows = []
    for row in rows:
        if should_remove(row, bad_paths, bad_ids):
            removed_rows.append(row)
        else:
            kept_rows.append(row)

    fieldnames = list(rows[0].keys())
    write_rows(args.out_labels, kept_rows, fieldnames)
    write_rows(args.removed_out, removed_rows, fieldnames)

    print(f"input rows: {len(rows)}")
    print(f"bad clip paths listed: {len(bad_paths)}")
    print(f"bad clip ids listed: {len(bad_ids)}")
    print(f"removed rows: {len(removed_rows)}")
    print(f"kept rows: {len(kept_rows)}")
    if reasons:
        print("manual reasons:")
        for reason, count in reasons.most_common():
            print(f"- {reason}: {count}")
    print(f"saved clean labels: {args.out_labels}")
    print(f"saved removed rows: {args.removed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
