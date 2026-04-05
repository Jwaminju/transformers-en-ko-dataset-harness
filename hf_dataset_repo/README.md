---
{
  "pretty_name": "Transformers EN-KO Aligned Docs",
  "language": [
    "en",
    "ko"
  ],
  "task_categories": [
    "translation"
  ],
  "tags": [
    "translation",
    "alignment",
    "korean",
    "english",
    "transformers-docs"
  ],
  "size_categories": [
    "n<1K"
  ]
}
---

# Transformers EN-KO Aligned Docs

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

- train rows: 76
- validation rows: 5
- stress-test rows: 880
- blacklist docs: 13

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
- source commit used locally: `ce7efd8f19`
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
