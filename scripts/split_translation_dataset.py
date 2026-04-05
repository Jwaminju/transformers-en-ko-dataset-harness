from __future__ import annotations

import argparse
import csv
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


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_doc_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "status", "github_url"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_github_url(relative_path: str) -> str:
    return f"https://github.com/huggingface/transformers/blob/main/docs/source/ko/{relative_path}"


def split_pairs(
    pair_rows: list[dict[str, object]],
    rejected_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, str]]]:
    contaminated_docs = {str(row["relative_path"]) for row in rejected_rows}
    train_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []

    all_docs = {str(row["relative_path"]) for row in pair_rows} | contaminated_docs
    doc_rows: list[dict[str, str]] = []

    for pair_row in pair_rows:
        relative_path = str(pair_row["relative_path"])
        if relative_path in contaminated_docs:
            eval_rows.append(pair_row)
        else:
            train_rows.append(pair_row)

    for relative_path in sorted(all_docs):
        doc_rows.append(
            {
                "relative_path": relative_path,
                "status": "contaminated" if relative_path in contaminated_docs else "clean",
                "github_url": build_github_url(relative_path),
            }
        )

    return train_rows, eval_rows, doc_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split translation pairs into clean-train and contaminated-eval sets.")
    parser.add_argument(
        "--pairs-input",
        default=str(Path("data") / "transformers_en_ko_pairs.jsonl"),
        help="Accepted pair JSONL from prepare_translation_dataset.py",
    )
    parser.add_argument(
        "--rejected-input",
        default=str(Path("data") / "transformers_en_ko_rejected_pairs.jsonl"),
        help="Rejected pair JSONL from prepare_translation_dataset.py",
    )
    parser.add_argument(
        "--train-output",
        default=str(Path("data") / "transformers_en_ko_train_clean.jsonl"),
        help="Clean-document training pairs",
    )
    parser.add_argument(
        "--eval-output",
        default=str(Path("data") / "transformers_en_ko_eval_contaminated.jsonl"),
        help="Contaminated-document evaluation pairs",
    )
    parser.add_argument(
        "--docs-output",
        default=str(Path("data") / "transformers_en_ko_doc_split.csv"),
        help="Document contamination status report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_input = Path(args.pairs_input)
    rejected_input = Path(args.rejected_input)
    train_output = Path(args.train_output)
    eval_output = Path(args.eval_output)
    docs_output = Path(args.docs_output)

    pair_rows = read_jsonl(pairs_input)
    rejected_rows = read_jsonl(rejected_input)
    train_rows, eval_rows, doc_rows = split_pairs(pair_rows, rejected_rows)

    train_output.parent.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_rows, train_output)
    write_jsonl(eval_rows, eval_output)
    write_doc_csv(doc_rows, docs_output)

    clean_docs = sum(1 for row in doc_rows if row["status"] == "clean")
    contaminated_docs = sum(1 for row in doc_rows if row["status"] == "contaminated")
    print(f"Wrote {len(train_rows)} clean training pairs to {train_output}")
    print(f"Wrote {len(eval_rows)} contaminated-document eval pairs to {eval_output}")
    print(f"Wrote {len(doc_rows)} document rows to {docs_output}")
    print(f"Clean docs: {clean_docs}")
    print(f"Contaminated docs: {contaminated_docs}")


if __name__ == "__main__":
    main()
