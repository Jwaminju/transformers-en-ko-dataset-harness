# Dataset Methodology

## Source

- upstream repository: `huggingface/transformers`
- source locales: `docs/source/en`, `docs/source/ko`
- source commit: `ce7efd8f19`

## Published splits

- `data/train.jsonl`: 76 rows
- `data/validation_clean.jsonl`: 5 rows
- `data/stress_test_contaminated.jsonl`: 880 rows
- `metadata/blacklist.txt`: 13 manually excluded documents

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
