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


def build_github_url(relative_path: str) -> str:
    return f"https://github.com/huggingface/transformers/blob/main/docs/source/ko/{relative_path}"


def truncate(text: str, max_chars: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def build_suspect_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["relative_path"])].append(row)

    suspect_rows: list[dict[str, object]] = []
    for relative_path, chunks in grouped.items():
        ratio_rows = []
        for chunk in chunks:
            source_chars = int(chunk["source_chars"])
            target_chars = max(int(chunk["target_chars"]), 1)
            ratio = source_chars / target_chars
            ratio_rows.append((ratio, chunk))

        ratio_rows.sort(key=lambda item: item[0], reverse=True)
        worst_ratio, worst_chunk = ratio_rows[0]
        avg_ratio = sum(ratio for ratio, _ in ratio_rows) / len(ratio_rows)

        suspect_rows.append(
            {
                "relative_path": relative_path,
                "github_url": build_github_url(relative_path),
                "chunks": str(len(chunks)),
                "max_ratio": f"{worst_ratio:.2f}",
                "avg_ratio": f"{avg_ratio:.2f}",
                "worst_chunk_id": str(worst_chunk["chunk_id"]),
                "worst_source_chars": str(worst_chunk["source_chars"]),
                "worst_target_chars": str(worst_chunk["target_chars"]),
                "worst_target_preview": truncate(str(worst_chunk["target"])),
            }
        )

    suspect_rows.sort(
        key=lambda row: (
            -float(row["max_ratio"]),
            -float(row["avg_ratio"]),
            row["relative_path"],
        )
    )
    return suspect_rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "relative_path",
        "github_url",
        "chunks",
        "max_ratio",
        "avg_ratio",
        "worst_chunk_id",
        "worst_source_chars",
        "worst_target_chars",
        "worst_target_preview",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean-doc suspect report ranked by source-target length ratios.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "transformers_en_ko_train_clean.jsonl"),
        help="Clean training pair JSONL",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "transformers_en_ko_clean_suspects.csv"),
        help="Suspect CSV output path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input))
    suspect_rows = build_suspect_rows(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(suspect_rows, output_path)
    print(f"Wrote {len(suspect_rows)} suspect rows to {output_path}")


if __name__ == "__main__":
    main()
