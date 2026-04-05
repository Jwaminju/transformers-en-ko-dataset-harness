from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_document(
    rows: list[dict[str, object]],
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["relative_path"])].append(row)

    paths = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(paths)

    validation_doc_count = max(1, round(len(paths) * validation_ratio))
    validation_paths = set(paths[:validation_doc_count])

    train_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for path in paths:
        target = validation_rows if path in validation_paths else train_rows
        target.extend(grouped[path])

    return train_rows, validation_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a clean training set into train/validation by document.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "transformers_en_ko_train_final.jsonl"),
        help="Clean final training JSONL",
    )
    parser.add_argument(
        "--train-output",
        default=str(Path("data") / "transformers_en_ko_train_split.jsonl"),
        help="Train split output",
    )
    parser.add_argument(
        "--validation-output",
        default=str(Path("data") / "transformers_en_ko_validation_clean.jsonl"),
        help="Validation split output",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    train_rows, validation_rows = split_by_document(
        rows,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, train_output)
    write_jsonl(validation_rows, validation_output)

    print(f"Wrote {len(train_rows)} rows to {train_output}")
    print(f"Wrote {len(validation_rows)} rows to {validation_output}")


if __name__ == "__main__":
    main()
