from __future__ import annotations

import argparse
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


def truncate_text(text: str, limit: int | None) -> str:
    compact = " ".join(str(text).split())
    if limit is None or len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_github_url(relative_path: str) -> str:
    return f"https://github.com/huggingface/transformers/blob/main/docs/source/ko/{relative_path}"


def group_rows(
    rows: list[dict[str, object]],
    max_docs: int,
    max_chunks_per_doc: int,
) -> list[tuple[str, list[dict[str, object]]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["relative_path"])].append(row)

    selected: list[tuple[str, list[dict[str, object]]]] = []
    for relative_path in sorted(grouped)[:max_docs]:
        chunks = sorted(grouped[relative_path], key=lambda row: int(row.get("chunk_id", 0)))
        selected.append((relative_path, chunks[:max_chunks_per_doc]))
    return selected


def build_markdown(
    rows: list[dict[str, object]],
    title: str,
    max_docs: int,
    max_chunks_per_doc: int,
    truncate_chars: int | None,
) -> str:
    lines = [f"# {title}", ""]
    grouped_rows = group_rows(rows, max_docs=max_docs, max_chunks_per_doc=max_chunks_per_doc)
    lines.append(f"- documents: {len(grouped_rows)}")
    lines.append(f"- chunks: {sum(len(chunks) for _, chunks in grouped_rows)}")
    lines.append("")

    for relative_path, chunks in grouped_rows:
        lines.append(f"## {relative_path}")
        lines.append(f"- link: {build_github_url(relative_path)}")
        lines.append("")
        for row in chunks:
            source_text = truncate_text(str(row["text"]), truncate_chars)
            target_text = truncate_text(str(row["target"]), truncate_chars)
            source_chars = len(str(row["text"]))
            target_chars = len(str(row["target"]))
            length_ratio = round(source_chars / max(target_chars, 1), 2)
            lines.append(f"### chunk {row['chunk_id']}")
            lines.append(f"- source_chars: {source_chars}")
            lines.append(f"- target_chars: {target_chars}")
            lines.append(f"- length_ratio: {length_ratio}")
            lines.append("")
            lines.append("**Source**")
            lines.append("")
            lines.append(source_text)
            lines.append("")
            lines.append("**Target**")
            lines.append("")
            lines.append(target_text)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readable Markdown review file for source-target alignment.")
    parser.add_argument(
        "--input",
        required=True,
        help="Pair JSONL file with text/target columns",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Markdown output path",
    )
    parser.add_argument("--title", default="Alignment Review")
    parser.add_argument("--max-docs", type=int, default=20)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--truncate-chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_jsonl(input_path)
    markdown = build_markdown(
        rows,
        title=args.title,
        max_docs=args.max_docs,
        max_chunks_per_doc=args.max_chunks_per_doc,
        truncate_chars=args.truncate_chars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote review markdown to {output_path}")


if __name__ == "__main__":
    main()
