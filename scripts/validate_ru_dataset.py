#!/usr/bin/env python3
"""Validate Russian lip reading dataset metadata and generate a short report."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_HEADER = [
    "clip_id",
    "clip_path",
    "word",
    "speaker_id",
    "source",
    "start_sec",
    "end_sec",
]


def read_vocab(vocab_path: Path) -> set[str]:
    if not vocab_path.exists():
        return set()
    words = set()
    for line in vocab_path.read_text(encoding="utf-8-sig").splitlines():
        token = line.strip()
        if token:
            words.add(token)
    return words


def normalize_fieldnames(fieldnames: list[str] | None) -> list[str] | None:
    if fieldnames is None:
        return None
    normalized = []
    for i, name in enumerate(fieldnames):
        if i == 0:
            normalized.append(name.lstrip("\ufeff"))
        else:
            normalized.append(name)
    return normalized


def validate(
    dataset_root: Path,
    min_per_word: int,
    labels_path: Path | None = None,
    vocab_path: Path | None = None,
) -> tuple[bool, str]:
    labels_path = labels_path or (dataset_root / "labels.csv")
    vocab_path = vocab_path or (dataset_root / "vocab.txt")

    problems: list[str] = []
    notes: list[str] = []

    if not labels_path.exists():
        return False, f"ERROR: missing labels file: {labels_path}"

    vocab = read_vocab(vocab_path)
    if not vocab:
        notes.append("WARN: vocab.txt is empty or missing words.")

    rows = []
    with labels_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        got_header = normalize_fieldnames(reader.fieldnames)
        if got_header != REQUIRED_HEADER:
            return False, (
                "ERROR: invalid header in labels.csv. "
                f"Expected: {REQUIRED_HEADER}, got: {got_header}"
            )
        for row in reader:
            rows.append(row)

    if not rows:
        notes.append("WARN: labels.csv has only header and no data rows yet.")
        report = [
            "Validation: PASS (template state)",
            "Rows: 0",
            "Words: 0",
            "Speakers: 0",
            *notes,
        ]
        return True, "\n".join(report)

    clip_ids = set()
    word_counter: Counter[str] = Counter()
    speaker_counter: Counter[str] = Counter()
    per_word_speakers: defaultdict[str, set[str]] = defaultdict(set)

    for i, row in enumerate(rows, start=2):
        clip_id = row["clip_id"].strip()
        clip_rel = row["clip_path"].strip()
        word = row["word"].strip()
        speaker = row["speaker_id"].strip()

        if not clip_id:
            problems.append(f"line {i}: empty clip_id")
            continue
        if clip_id in clip_ids:
            problems.append(f"line {i}: duplicate clip_id '{clip_id}'")
        clip_ids.add(clip_id)

        clip_abs = dataset_root / clip_rel
        if not clip_abs.exists():
            problems.append(f"line {i}: missing clip file '{clip_rel}'")

        if vocab and word not in vocab:
            problems.append(f"line {i}: word '{word}' not found in vocab.txt")

        try:
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if end_sec <= start_sec:
                problems.append(f"line {i}: end_sec <= start_sec")
        except ValueError:
            problems.append(f"line {i}: start_sec/end_sec are not valid floats")

        word_counter[word] += 1
        speaker_counter[speaker] += 1
        per_word_speakers[word].add(speaker)

    for word, count in sorted(word_counter.items()):
        if count < min_per_word:
            notes.append(f"WARN: word '{word}' has only {count} clips (< {min_per_word})")
        if len(per_word_speakers[word]) < 2:
            notes.append(f"WARN: word '{word}' has data from only 1 speaker")

    status = "PASS" if not problems else "FAIL"
    report = [
        f"Validation: {status}",
        f"Rows: {len(rows)}",
        f"Words: {len(word_counter)}",
        f"Speakers: {len(speaker_counter)}",
        "",
        "Per-word counts:",
    ]
    report.extend(f"- {w}: {c}" for w, c in sorted(word_counter.items()))

    if notes:
        report.append("")
        report.append("Notes:")
        report.extend(f"- {n}" for n in notes)

    if problems:
        report.append("")
        report.append("Problems:")
        report.extend(f"- {p}" for p in problems)

    return not problems, "\n".join(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ru_dataset metadata")
    parser.add_argument("--dataset-root", type=Path, default=Path("ru_dataset"))
    parser.add_argument("--min-per-word", type=int, default=15)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    ok, report = validate(
        args.dataset_root,
        args.min_per_word,
        labels_path=args.labels,
        vocab_path=args.vocab,
    )
    print(report)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved report: {args.report}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
