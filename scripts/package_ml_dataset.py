"""Build a portable ZIP archive from selected parts of one ML split.

Only clips referenced by the split CSV files are included. Archive member paths
always use forward slashes, so the result works in both Colab and Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", default="ru_dataset")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Split CSV files to include. Use '--splits train val' while tuning without test data.",
    )
    parser.add_argument(
        "--include-tool",
        type=Path,
        action="append",
        default=[],
        help="Optional local utility to include under tools/ for a Colab handoff.",
    )
    return parser.parse_args()


def read_split_rows(
    split_dir: Path,
    split_names: list[str] | tuple[str, ...],
) -> tuple[dict[str, list[dict[str, str]]], list[Path]]:
    split_rows: dict[str, list[dict[str, str]]] = {}
    clip_paths: list[Path] = []

    for split_name in split_names:
        csv_path = split_dir / f"{split_name}.csv"
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        split_rows[split_name] = rows
        clip_paths.extend(Path(row["clip_path"]) for row in rows)

    return split_rows, clip_paths


def archive_name(archive_root: str, relative_path: Path) -> str:
    return str(PurePosixPath(archive_root, *relative_path.parts))


def validate_relative_path(path: Path) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"clip_path must stay inside dataset root: {path}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    split_dir = args.split_dir.resolve()
    output = args.output.resolve()

    split_rows, clip_paths = read_split_rows(split_dir, args.splits)
    unique_clip_paths = sorted(set(clip_paths), key=lambda path: path.as_posix())
    for clip_path in unique_clip_paths:
        validate_relative_path(clip_path)
    missing = [path for path in unique_clip_paths if not (dataset_root / path).is_file()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} referenced clips are missing under {dataset_root}:\n{preview}"
        )
    missing_tools = [path for path in args.include_tool if not path.is_file()]
    if missing_tools:
        raise FileNotFoundError(f"Missing included tools: {missing_tools}")

    output.parent.mkdir(parents=True, exist_ok=True)
    split_relative = split_dir.relative_to(dataset_root)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for clip_path in unique_clip_paths:
            archive.write(
                dataset_root / clip_path,
                archive_name(args.archive_root, clip_path),
            )

        metadata_names = [f"{name}.csv" for name in args.splits] + ["vocab.txt", "README.md"]
        for metadata_name in metadata_names:
            metadata_path = split_dir / metadata_name
            if metadata_path.is_file():
                archive.write(
                    metadata_path,
                    archive_name(args.archive_root, split_relative / metadata_name),
                )

        for tool_path in args.include_tool:
            archive.write(tool_path, str(PurePosixPath("tools", tool_path.name)))

    split_counts = {name: len(rows) for name, rows in split_rows.items()}
    print("archive:", output)
    print("split counts:", split_counts)
    print("unique clips:", len(unique_clip_paths))
    print("size MB:", round(output.stat().st_size / 1024 / 1024, 2))


if __name__ == "__main__":
    main()
