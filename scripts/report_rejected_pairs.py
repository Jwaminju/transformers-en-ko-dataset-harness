from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReportRow:
    relative_path: str
    github_url: str
    rejected_pairs: int
    high_english_ratio: int
    mostly_english_target: int
    length_ratio: int
    sample_target: str

    def as_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "github_url": self.github_url,
            "rejected_pairs": self.rejected_pairs,
            "high_english_ratio": self.high_english_ratio,
            "mostly_english_target": self.mostly_english_target,
            "length_ratio": self.length_ratio,
            "sample_target": self.sample_target,
        }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate_text(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def build_report_rows(rows: list[dict[str, object]]) -> list[ReportRow]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["relative_path"])].append(row)

    report_rows: list[ReportRow] = []
    for relative_path, group in grouped.items():
        counts = Counter(str(item["reason"]) for item in group)
        english_first = next(
            (
                item
                for item in group
                if str(item["reason"]) in {"high_english_ratio", "mostly_english_target"}
            ),
            group[0],
        )
        report_rows.append(
            ReportRow(
                relative_path=relative_path,
                github_url=str(group[0]["github_url"]),
                rejected_pairs=len(group),
                high_english_ratio=counts["high_english_ratio"],
                mostly_english_target=counts["mostly_english_target"],
                length_ratio=counts["length_ratio"],
                sample_target=truncate_text(str(english_first["target"])),
            )
        )

    report_rows.sort(
        key=lambda row: (
            -row.high_english_ratio - row.mostly_english_target,
            -row.rejected_pairs,
            row.relative_path,
        )
    )
    return report_rows


def write_csv(rows: list[ReportRow], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "github_url",
                "rejected_pairs",
                "high_english_ratio",
                "mostly_english_target",
                "length_ratio",
                "sample_target",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_json(rows: list[ReportRow], output_path: Path) -> None:
    payload = [row.as_dict() for row in rows]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize rejected translation pairs by document.")
    parser.add_argument(
        "--input",
        default=str(Path("data") / "transformers_en_ko_rejected_pairs.jsonl"),
        help="Rejected pair JSONL from prepare_translation_dataset.py",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path("data") / "transformers_en_ko_rejected_docs.csv"),
        help="CSV report path",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path("data") / "transformers_en_ko_rejected_docs.json"),
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    rows = read_jsonl(input_path)
    report_rows = build_report_rows(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_csv(report_rows, output_csv)
    write_json(report_rows, output_json)

    print(f"Wrote {len(report_rows)} rejected-doc rows to {output_csv}")
    print(f"Wrote {len(report_rows)} rejected-doc rows to {output_json}")


if __name__ == "__main__":
    main()
