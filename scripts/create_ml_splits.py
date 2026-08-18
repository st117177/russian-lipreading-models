#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


DEFAULT_TRAIN_SPEAKERS = ["spk03", "spk04", "spk05", "spk06"]
DEFAULT_VAL_SPEAKERS = ["spk09", "spk10"]
DEFAULT_TEST_SPEAKERS = ["spk07", "spk08"]
DEFAULT_TOP_WORDS = 10

def parse_speakers(value: str) -> set[str]:
    return {speaker.strip() for speaker in value.split(",") if speaker.strip()}

# Read CSV rows from a file and return them as a list of dictionaries.
def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

# Write a list of dictionaries to a CSV file with the specified fieldnames.
def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Return the top n most common words from the rows.
def top_words(rows: list[dict[str, str]], n: int) -> list[str]:
    counts = Counter(row["word"] for row in rows)
    return [word for word, _ in counts.most_common(n)]

# Filter the rows to include only those with words in the specified set.
def filter_words(rows: list[dict[str, str]], words: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row["word"] in words]

# Split the rows into train, val, and test sets based on speaker IDs.
def speaker_split(
    rows: list[dict[str, str]],
    #rows — это list; внутри list лежат dict
    # у каждого dict ключи str и значения str
    train_speakers: set[str],
    val_speakers: set[str],
    test_speakers: set[str],
) -> dict[str, list[dict[str, str]]]:
    # Возвращаем словарь из трёх частей:
    # {
    #   "train": [...строки train...],
    #   "val": [...строки val...],
    #   "test": [...строки test...]
    # }
    return {
        # В train берём только те строки, где speaker_id входит в train_speakers.
        "train": [row for row in rows if row["speaker_id"] in train_speakers],

        # В val берём только те строки, где speaker_id входит в val_speakers.
        "val": [row for row in rows if row["speaker_id"] in val_speakers],

        # В test берём только те строки, где speaker_id входит в test_speakers.
        "test": [row for row in rows if row["speaker_id"] in test_speakers],
    }

# Split the rows into train, val, and test sets in a stratified random manner based on words.
def stratified_random_split(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    by_word: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_word.setdefault(row["word"], []).append(row)

    splits = {"train": [], "val": [], "test": []}
    for word_rows in by_word.values():
        shuffled = list(word_rows)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n > 2 else 0
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    for split_rows in splits.values():
        split_rows.sort(key=lambda row: row["clip_id"])
    return splits

# Generate a description of the split, including counts of rows, words, and speakers.
def describe(name: str, rows: list[dict[str, str]]) -> list[str]:
    word_counts = Counter(row["word"] for row in rows)
    speaker_counts = Counter(row["speaker_id"] for row in rows)
    lines = [
        f"## {name}",
        f"rows: {len(rows)}",
        f"words: {len(word_counts)}",
        f"speakers: {len(speaker_counts)}",
        "",
        "speakers:",
    ]
    lines.extend(f"- {speaker}: {count}" for speaker, count in sorted(speaker_counts.items()))
    lines.append("")
    lines.append("words:")
    lines.extend(f"- {word}: {count}" for word, count in word_counts.most_common())
    return lines

# whites README.md file for the split set, including counts of rows, words, and speakers.
def write_manifest(
    path: Path,
    split_name: str,
    splits: dict[str, list[dict[str, str]]],
    source_labels: Path,
    words: list[str] | None,
) -> None:
    lines = [
        f"# {split_name}",
        "",
        f"Files in this folder are derived from {source_labels.as_posix()}.",
        "",
    ]
    if words is not None:
        lines.append("Vocabulary:")
        lines.extend(f"- {word}" for word in words)
        lines.append("")
    for key in ["train", "val", "test"]:
        lines.extend(describe(key, splits[key]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

# Save the split sets to CSV files and write a manifest file.
def save_split_set(
    out_dir: Path,
    split_name: str,
    splits: dict[str, list[dict[str, str]]],
    fieldnames: list[str],
    source_labels: Path,
    words: list[str] | None = None,
) -> None:
    split_dir = out_dir / split_name
    for key, split_rows in splits.items():
        write_rows(split_dir / f"{key}.csv", split_rows, fieldnames)
    if words is not None:
        (split_dir / "vocab.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    write_manifest(split_dir / "README.md", split_name, splits, source_labels, words)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create ML-ready split CSV files.")
    parser.add_argument("--labels", type=Path, default=Path("ru_dataset/selected_labels.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("ru_dataset/ml_splits"))
    parser.add_argument("--top-words", type=int, default=DEFAULT_TOP_WORDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-speakers", default=",".join(DEFAULT_TRAIN_SPEAKERS))
    parser.add_argument("--val-speakers", default=",".join(DEFAULT_VAL_SPEAKERS))
    parser.add_argument("--test-speakers", default=",".join(DEFAULT_TEST_SPEAKERS))
    parser.add_argument("--exclude-speakers", default="")
    args = parser.parse_args()

    rows = read_rows(args.labels)
    if not rows:
        raise SystemExit("No rows found in labels CSV")
    exclude_speakers = parse_speakers(args.exclude_speakers)
    if exclude_speakers:
        rows = [row for row in rows if row["speaker_id"] not in exclude_speakers]
        if not rows:
            raise SystemExit("No rows left after speaker exclusion")
    fieldnames = list(rows[0].keys())

    train_speakers = parse_speakers(args.train_speakers)
    val_speakers = parse_speakers(args.val_speakers)
    test_speakers = parse_speakers(args.test_speakers)

    full_speaker = speaker_split(rows, train_speakers, val_speakers, test_speakers)
    save_split_set(args.out_dir, "speaker_full14", full_speaker, fieldnames, args.labels)

    top = top_words(rows, args.top_words)
    top_rows = filter_words(rows, set(top))
    top_speaker = speaker_split(top_rows, train_speakers, val_speakers, test_speakers)
    save_split_set(args.out_dir, f"speaker_top{args.top_words}", top_speaker, fieldnames, args.labels, top)

    random_full = stratified_random_split(rows, args.seed, train_ratio=0.8, val_ratio=0.1)
    save_split_set(args.out_dir, "random_full14", random_full, fieldnames, args.labels)

    random_top = stratified_random_split(top_rows, args.seed, train_ratio=0.8, val_ratio=0.1)
    save_split_set(args.out_dir, f"random_top{args.top_words}", random_top, fieldnames, args.labels, top)

    print(f"Saved splits to {args.out_dir}")
    print(f"Top-{args.top_words} words: {', '.join(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
