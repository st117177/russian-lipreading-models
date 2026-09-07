"""Audit overlapping padded word intervals without modifying the dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def find_overlaps(manifest: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest)
    required = {
        "clip_file", "word", "speaker_id", "start_frame", "end_frame",
        "start_sec", "end_sec", "source_video",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{manifest} is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for source_video, group in frame.groupby("source_video", sort=True):
        ordered = group.sort_values(["start_sec", "end_sec", "clip_file"])
        records = ordered.to_dict("records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                if float(right["start_sec"]) >= float(left["end_sec"]):
                    break
                overlap_start = max(float(left["start_sec"]), float(right["start_sec"]))
                overlap_end = min(float(left["end_sec"]), float(right["end_sec"]))
                if overlap_end <= overlap_start or left["word"] == right["word"]:
                    continue
                rows.append(
                    {
                        "source_video": source_video,
                        "speaker_id": left["speaker_id"],
                        "word_a": left["word"],
                        "word_b": right["word"],
                        "clip_id_a": Path(str(left["clip_file"])).stem,
                        "clip_id_b": Path(str(right["clip_file"])).stem,
                        "clip_file_a": left["clip_file"],
                        "clip_file_b": right["clip_file"],
                        "start_sec_a": left["start_sec"],
                        "end_sec_a": left["end_sec"],
                        "start_sec_b": right["start_sec"],
                        "end_sec_b": right["end_sec"],
                        "overlap_sec": round(overlap_end - overlap_start, 6),
                        "overlap_start_sec": round(overlap_start, 6),
                        "overlap_end_sec": round(overlap_end, 6),
                    }
                )

    columns = [
        "source_video", "speaker_id", "word_a", "word_b", "clip_id_a", "clip_id_b",
        "clip_file_a", "clip_file_b", "start_sec_a", "end_sec_a", "start_sec_b",
        "end_sec_b", "overlap_sec", "overlap_start_sec", "overlap_end_sec",
    ]
    return pd.DataFrame(rows, columns=columns)


def select_review_sample(overlaps: pd.DataFrame, max_items: int) -> pd.DataFrame:
    if overlaps.empty:
        return overlaps.copy()
    ordered = overlaps.sort_values(
        ["speaker_id", "overlap_sec", "source_video"],
        ascending=[True, False, True],
    )
    selected: list[int] = []
    per_speaker = max(1, max_items // max(1, ordered["speaker_id"].nunique()))
    for _, group in ordered.groupby("speaker_id", sort=True):
        selected.extend(group.head(per_speaker).index.tolist())
    if len(selected) < max_items:
        selected.extend(index for index in ordered.index if index not in selected)
    sample = ordered.loc[selected[:max_items]].reset_index(drop=True)
    sample["review_status"] = "pending"
    sample["notes"] = ""
    return sample


def write_report(manifest: Path, overlaps: pd.DataFrame, sample: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Padded interval overlap audit", "",
        "This is a read-only audit. It does not delete or rewrite clips.", "",
        f"Manifest: `{manifest}`",
        f"Total different-word overlap pairs: **{len(overlaps)}**",
        f"Selected for manual review: **{len(sample)}**", "",
        "## Interpretation", "",
        "Padding can create an overlap even when original word boundaries were correct. "
        "Review only checks whether neighbouring articulation is visible in the final crop.", "",
    ]
    if overlaps.empty:
        lines.append("No different-word overlaps were found.")
    else:
        lines.extend(["## Pairs by speaker", ""])
        lines.extend(
            f"- `{speaker}`: {count}"
            for speaker, count in overlaps.groupby("speaker_id").size().sort_index().items()
        )
        lines.extend(["", "## Review rule", ""])
        lines.append(
            "Mark `keep` when the target word is isolated and `inspect` when a neighbour "
            "is visibly present. Only confirmed bad boundaries should be excluded later."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find different-word overlaps in padded manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()

    overlaps = find_overlaps(args.manifest)
    sample = select_review_sample(overlaps, args.sample_size)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    overlaps.to_csv(args.out_csv, index=False)
    sample.to_csv(args.sample_csv, index=False)
    write_report(args.manifest, overlaps, sample, args.report)
    print(f"manifest: {args.manifest}")
    print(f"overlap pairs: {len(overlaps)}")
    print(f"review sample: {len(sample)}")
    print(f"saved: {args.out_csv}")
    print(f"saved: {args.sample_csv}")
    print(f"saved: {args.report}")


if __name__ == "__main__":
    main()
