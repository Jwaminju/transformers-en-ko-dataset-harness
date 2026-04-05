---
name: translation-dataset-build
description: Use when rebuilding or updating the EN-to-KO translation fine-tuning dataset from multilingual documentation, including alignment, filtering, review artifact generation, and Hugging Face dataset packaging.
---

# Translation Dataset Build

Use this skill when the user wants to regenerate the translation dataset, improve the filtering pipeline, or package the resulting splits for Hugging Face.

## Read First

- [AGENTS.md](../../../AGENTS.md)
- [translation_dataset_harness.md](../../../docs/translation_dataset_harness.md)
- [README.md](../../../README.md)

## Core Rules

- Same relative path does not imply a valid translation pair.
- Prefer conservative rejection over forced alignment.
- Keep alignment-time preprocessing separate from training-time cleanup.
- Treat `data/transformers_en_ko_eval_contaminated.jsonl` as stress test data, not the main validation set.
- Do not upload if required review artifacts or package metadata are missing.

## Canonical Workflow

1. Build document pairs with `scripts/build_translation_corpus.py`.
2. Build aligned chunk pairs with `scripts/prepare_translation_dataset.py`.
3. Summarize rejects with `scripts/report_rejected_pairs.py`.
4. Split documents into clean vs contaminated with `scripts/split_translation_dataset.py`.
5. Build review surfaces with `scripts/build_alignment_review.py` and `scripts/build_suspect_review.py`.
6. Apply `data/blacklist.txt` with `scripts/filter_blacklist_pairs.py`.
7. Create train/validation splits with `scripts/split_train_validation.py`.
8. Package the dataset repo with `scripts/build_hf_dataset_repo.py`.
9. Upload only after package gates pass with `scripts/upload_hf_dataset.py`.

## Common Commands

Rebuild the dataset package from the final splits:

```bash
uv run python scripts/build_hf_dataset_repo.py \
  --train-input data/transformers_en_ko_train_split.jsonl \
  --validation-input data/transformers_en_ko_validation_clean.jsonl \
  --eval-input data/transformers_en_ko_eval_contaminated.jsonl \
  --blacklist-input data/blacklist.txt \
  --doc-split-input data/transformers_en_ko_doc_split.csv \
  --rejected-docs-input data/transformers_en_ko_rejected_docs.csv \
  --source-commit ce7efd8f19 \
  --dataset-name "Transformers EN-KO Aligned Docs" \
  --output-dir hf_dataset_repo
```

Upload the packaged dataset only after the package status is `passed`:

```bash
uv run python scripts/upload_hf_dataset.py \
  --repo-id jmj-minju/transformers-en-ko-aligned-docs \
  --folder hf_dataset_repo \
  --private
```

Rebuild clean train and validation splits before packaging:

```bash
uv run python scripts/split_train_validation.py \
  --input data/transformers_en_ko_train_final.jsonl \
  --train-output data/transformers_en_ko_train_split.jsonl \
  --validation-output data/transformers_en_ko_validation_clean.jsonl \
  --validation-ratio 0.1 \
  --seed 42
```

## Required Artifacts

Do not consider the run complete unless these exist:

- `data/transformers_en_ko_train_split.jsonl`
- `data/transformers_en_ko_validation_clean.jsonl`
- `data/transformers_en_ko_eval_contaminated.jsonl`
- `data/blacklist.txt`
- `data/review/train_clean_review.md`
- `data/review/eval_contaminated_review.md`
- `data/review/clean_suspect_review.md`
- `hf_dataset_repo/metadata/build_status.json`

## Review and Escalation

- Let the agent triage first.
- Route only uncertain or mixed cases to human review.
- If the first chunk of a document is clearly rewritten or structurally mismatched, drop the document.
- If headings or code blocks appear to be serving as anchors, preserve them for alignment analysis instead of stripping them blindly.

## Packaging Policy

- `hf_dataset_repo/data/` should contain only published splits.
- `hf_dataset_repo/metadata/` should explain filtering decisions.
- `hf_dataset_repo/docs/` should preserve the harness and dataset methodology.

## When to Stop

Stop and ask for confirmation if:

- warning thresholds in the harness are tripped
- package status is not `passed`
- a change would reduce the clean set sharply without a clear reason
