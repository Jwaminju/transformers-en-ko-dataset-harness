from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_blacklist(path: Path) -> set[str]:
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.add(line)
    return entries


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter translation pairs by a blacklist of relative paths.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "transformers_en_ko_train_clean.jsonl"),
        help="Input pair JSONL",
    )
    parser.add_argument(
        "--blacklist",
        default=str(Path("data") / "blacklist.txt"),
        help="Blacklist file with one relative path per line",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "transformers_en_ko_train_final.jsonl"),
        help="Filtered output pair JSONL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    blacklist = read_blacklist(Path(args.blacklist))
    filtered = [row for row in rows if str(row["relative_path"]) not in blacklist]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(filtered, output_path)

    print(f"Wrote {len(filtered)} filtered training pairs to {output_path}")
    print(f"Blacklisted docs: {len(blacklist)}")


if __name__ == "__main__":
    main()
