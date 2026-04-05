from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def count_blacklist_rows(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            count += 1
    return count


def build_readme(
    train_rows: int,
    eval_rows: int,
    validation_rows: int,
    blacklist_rows: int,
    source_commit: str,
    dataset_name: str,
) -> str:
    metadata = {
        "pretty_name": dataset_name,
        "language": ["en", "ko"],
        "task_categories": ["translation"],
        "tags": ["translation", "alignment", "korean", "english", "transformers-docs"],
        "size_categories": ["n<1K"],
    }
    metadata_block = "---\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n---"

    return f"""{metadata_block}

# {dataset_name}

This dataset package contains English-Korean aligned text pairs derived from the `docs/source/en` and `docs/source/ko` trees in [`huggingface/transformers`](https://github.com/huggingface/transformers).

## Repository layout

- `data/`: published dataset splits only
- `metadata/`: filtering, blacklist, and build-status artifacts
- `docs/`: agent harness and dataset construction notes
- `AGENTS.md`: short Codex entry point for this dataset repo

## Contents

- `data/train.jsonl`: final training split with manually reviewed blacklist applied
- `data/validation_clean.jsonl`: clean validation split sampled by document from the final training set
- `data/stress_test_contaminated.jsonl`: document pairs excluded from training because they contained alignment or contamination issues
- `metadata/blacklist.txt`: documents manually excluded after review
- `metadata/doc_split.csv`: document-level clean vs contaminated split
- `metadata/rejected_docs.csv`: rejected-document summary with reasons
- `metadata/build_status.json`: machine-readable build summary
- `docs/translation_dataset_harness.md`: detailed harness and review policy
- `docs/dataset_methodology.md`: concise build and usage notes

## Row counts

- train rows: {train_rows}
- validation rows: {validation_rows}
- stress-test rows: {eval_rows}
- blacklist docs: {blacklist_rows}

## Schema

Each JSONL row contains:

- `relative_path`
- `chunk_id`
- `text`
- `target`
- `source_chars`
- `target_chars`

## Build notes

- source repository: `huggingface/transformers`
- source commit used locally: `{source_commit}`
- alignment method: markdown normalization, one-sided metadata removal, conservative block-level alignment, document rejection for structural mismatch, and manual blacklist
- preprocessing note: the current exported dataset was built with an earlier cleanup-heavy pipeline, but the harness now recommends preserving headings and code-block structure during alignment and cleaning more aggressively only at export time

## Intended use

- `train.jsonl` is the conservative split intended for fine-tuning.
- `validation_clean.jsonl` is the clean validation split for model selection.
- `stress_test_contaminated.jsonl` is not a gold benchmark. It is a stress-test set containing documents that were rejected from training because alignment looked unreliable.

## Limitations

- The dataset is partially automatically aligned and then manually filtered.
- Some original docs are localized or rewritten rather than directly translated, and those files should not be treated as parallel data.
- Review the blacklist and rejected reports before using this dataset for evaluation claims.

## Source and attribution

Original documentation content comes from the Hugging Face Transformers repository:

- English: `docs/source/en`
- Korean: `docs/source/ko`

Before publishing publicly, verify that your redistribution plan matches the upstream repository license and your intended use.
"""


def build_agents_md() -> str:
    return """# Dataset Repo Entry Point

Read these first:

- `README.md`
- `docs/translation_dataset_harness.md`
- `docs/dataset_methodology.md`

Rules:

- Treat `data/` as the published artifact surface.
- Treat `metadata/` as the explanation layer for filtering and rejection decisions.
- `stress_test_contaminated.jsonl` is not the main validation set.
- If you rebuild this dataset elsewhere, keep review artifacts and blacklist decisions versioned.
"""


def build_methodology(
    train_rows: int,
    eval_rows: int,
    validation_rows: int,
    blacklist_rows: int,
    source_commit: str,
) -> str:
    return f"""# Dataset Methodology

## Source

- upstream repository: `huggingface/transformers`
- source locales: `docs/source/en`, `docs/source/ko`
- source commit: `{source_commit}`

## Published splits

- `data/train.jsonl`: {train_rows} rows
- `data/validation_clean.jsonl`: {validation_rows} rows
- `data/stress_test_contaminated.jsonl`: {eval_rows} rows
- `metadata/blacklist.txt`: {blacklist_rows} manually excluded documents

## Build approach

1. Match source and target files by relative path.
2. Normalize markdown and remove obvious one-sided artifacts.
3. Align blocks conservatively.
4. Reject documents with structural mismatch.
5. Reject chunks with contamination or severe drift.
6. Review suspicious cases and apply a manual blacklist.
7. Export clean train and validation splits plus a contaminated stress-test set.

## Important caveat

Same relative path does not imply a valid translation pair. Some documentation files are localized or rewritten, not directly translated. Those files belong in rejection metadata or stress testing, not in the main training split.

## Recommended usage

- Use `data/train.jsonl` for fine-tuning.
- Use `data/validation_clean.jsonl` for model selection.
- Use `data/stress_test_contaminated.jsonl` only for robustness checks or qualitative stress testing.

## Review references

- `metadata/blacklist.txt`
- `metadata/doc_split.csv`
- `metadata/rejected_docs.csv`
- `docs/translation_dataset_harness.md`
"""


def build_status(
    train_rows: int,
    eval_rows: int,
    validation_rows: int,
    blacklist_rows: int,
    source_commit: str,
) -> dict[str, int | str]:
    return {
        "source_commit": source_commit,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "stress_test_rows": eval_rows,
        "blacklist_docs": blacklist_rows,
        "status": "passed",
    }


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Hugging Face dataset repository folder from local outputs.")
    parser.add_argument("--train-input", default="data/transformers_en_ko_train_final.jsonl")
    parser.add_argument("--eval-input", default="data/transformers_en_ko_eval_contaminated.jsonl")
    parser.add_argument("--validation-input", default="data/transformers_en_ko_validation_clean.jsonl")
    parser.add_argument("--blacklist-input", default="data/blacklist.txt")
    parser.add_argument("--doc-split-input", default="data/transformers_en_ko_doc_split.csv")
    parser.add_argument("--rejected-docs-input", default="data/transformers_en_ko_rejected_docs.csv")
    parser.add_argument("--harness-input", default="docs/translation_dataset_harness.md")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--dataset-name", default="Transformers EN-KO Aligned Docs")
    parser.add_argument("--output-dir", default="hf_dataset_repo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_input = Path(args.train_input)
    eval_input = Path(args.eval_input)
    validation_input = Path(args.validation_input)
    blacklist_input = Path(args.blacklist_input)
    doc_split_input = Path(args.doc_split_input)
    rejected_docs_input = Path(args.rejected_docs_input)
    harness_input = Path(args.harness_input)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    data_dir = output_dir / "data"
    metadata_dir = output_dir / "metadata"
    docs_dir = output_dir / "docs"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    train_rows = count_jsonl_rows(train_input)
    eval_rows = count_jsonl_rows(eval_input)
    validation_rows = count_jsonl_rows(validation_input)
    blacklist_rows = count_blacklist_rows(blacklist_input)

    copy_file(train_input, data_dir / "train.jsonl")
    copy_file(validation_input, data_dir / "validation_clean.jsonl")
    copy_file(eval_input, data_dir / "stress_test_contaminated.jsonl")
    copy_file(blacklist_input, metadata_dir / "blacklist.txt")
    copy_file(doc_split_input, metadata_dir / "doc_split.csv")
    copy_file(rejected_docs_input, metadata_dir / "rejected_docs.csv")
    if harness_input.exists():
        copy_file(harness_input, docs_dir / "translation_dataset_harness.md")

    (docs_dir / "dataset_methodology.md").write_text(
        build_methodology(
            train_rows=train_rows,
            eval_rows=eval_rows,
            validation_rows=validation_rows,
            blacklist_rows=blacklist_rows,
            source_commit=args.source_commit,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "build_status.json").write_text(
        json.dumps(
            build_status(
                train_rows=train_rows,
                eval_rows=eval_rows,
                validation_rows=validation_rows,
                blacklist_rows=blacklist_rows,
                source_commit=args.source_commit,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "AGENTS.md").write_text(build_agents_md(), encoding="utf-8")

    readme = build_readme(
        train_rows=train_rows,
        eval_rows=eval_rows,
        validation_rows=validation_rows,
        blacklist_rows=blacklist_rows,
        source_commit=args.source_commit,
        dataset_name=args.dataset_name,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Built dataset repo folder at {output_dir}")


if __name__ == "__main__":
    main()
