from __future__ import annotations

import argparse
import csv
import json
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def truncate_text(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_markdown(
    pair_rows: list[dict[str, object]],
    suspect_rows: list[dict[str, str]],
    max_docs: int,
    max_chunks_per_doc: int,
    truncate_chars: int,
) -> str:
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        by_path[str(row["relative_path"])].append(row)

    lines = ["# Clean Suspect Review", ""]
    selected = suspect_rows[:max_docs]
    lines.append(f"- documents: {len(selected)}")
    lines.append("")

    for suspect in selected:
        relative_path = suspect["relative_path"]
        lines.append(f"## {relative_path}")
        lines.append(f"- link: {suspect['github_url']}")
        lines.append(f"- max_ratio: {suspect['max_ratio']}")
        lines.append(f"- avg_ratio: {suspect['avg_ratio']}")
        lines.append("")

        rows = sorted(
            by_path.get(relative_path, []),
            key=lambda row: row["source_chars"] / max(row["target_chars"], 1),
            reverse=True,
        )[:max_chunks_per_doc]

        for row in rows:
            ratio = round(row["source_chars"] / max(row["target_chars"], 1), 2)
            lines.append(f"### chunk {row['chunk_id']}")
            lines.append(f"- ratio: {ratio}")
            lines.append(f"- source_chars: {row['source_chars']}")
            lines.append(f"- target_chars: {row['target_chars']}")
            lines.append("")
            lines.append("**Source**")
            lines.append("")
            lines.append(truncate_text(str(row["text"]), truncate_chars))
            lines.append("")
            lines.append("**Target**")
            lines.append("")
            lines.append(truncate_text(str(row["target"]), truncate_chars))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a focused Markdown review for suspicious clean documents.")
    parser.add_argument(
        "--pairs-input",
        default=str(Path("data") / "transformers_en_ko_train_clean.jsonl"),
        help="Clean training pair JSONL",
    )
    parser.add_argument(
        "--suspects-input",
        default=str(Path("data") / "transformers_en_ko_clean_suspects.csv"),
        help="CSV generated from clean training pairs",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "review" / "clean_suspect_review.md"),
        help="Markdown output path",
    )
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--max-chunks-per-doc", type=int, default=2)
    parser.add_argument("--truncate-chars", type=int, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = read_jsonl(Path(args.pairs_input))
    suspect_rows = read_csv(Path(args.suspects_input))

    markdown = build_markdown(
        pair_rows,
        suspect_rows,
        max_docs=args.max_docs,
        max_chunks_per_doc=args.max_chunks_per_doc,
        truncate_chars=args.truncate_chars,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote suspect review markdown to {output_path}")


if __name__ == "__main__":
    main()
