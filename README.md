# Transformers EN-KO Dataset Harness

Codex-oriented source repository for building a conservative English-to-Korean translation fine-tuning dataset from multilingual technical documentation.

The current target corpus comes from `huggingface/transformers/docs/source/en` and `docs/source/ko`, but the harness is designed around a more general rule:

- same relative path does not imply a valid translation pair
- alignment must be conservative
- rewritten or localized documents should be rejected instead of forced into the training set

For the full operating policy, read [docs/translation_dataset_harness.md](docs/translation_dataset_harness.md).

## Repository layout

- `docs/`: harness and methodology
- `.codex/`: runtime hooks and guardrails
- `.agents/skills/`: repo-scoped build and review workflows
- `scripts/`: dataset build, review, packaging, and upload commands
- `tests/`: unit tests for the translation harness
- `data/`: generated intermediate, final, and review artifacts
- `hf_dataset_repo/`: upload-ready Hugging Face dataset package

## Quick start

Install dependencies:

```bash
uv sync --extra translation --extra dev
```

Build the document-level corpus:

```bash
uv run python scripts/build_translation_corpus.py \
  --repo-url https://github.com/huggingface/transformers.git \
  --repo-dir .cache/huggingface-transformers \
  --output data/transformers_en_ko_docs.jsonl
```

Build aligned chunk pairs:

```bash
uv run python scripts/prepare_translation_dataset.py \
  --input data/transformers_en_ko_docs.jsonl \
  --output data/transformers_en_ko_pairs.jsonl
```

Split clean vs contaminated docs:

```bash
uv run python scripts/split_translation_dataset.py \
  --pairs-input data/transformers_en_ko_pairs.jsonl \
  --rejected-input data/transformers_en_ko_rejected_pairs.jsonl \
  --train-output data/transformers_en_ko_train_clean.jsonl \
  --eval-output data/transformers_en_ko_eval_contaminated.jsonl \
  --docs-output data/transformers_en_ko_doc_split.csv
```

Build the upload package:

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

Upload the dataset package:

```bash
uv run python scripts/upload_hf_dataset.py \
  --repo-id jmj-minju/transformers-en-ko-aligned-docs \
  --folder hf_dataset_repo \
  --private
```

## Fine-tuning

Train on the clean split and validate on the clean validation set:

```bash
uv run python scripts/train_translation_model.py \
  --train-file data/transformers_en_ko_train_split.jsonl \
  --validation-file data/transformers_en_ko_validation_clean.jsonl \
  --model-name-or-path Helsinki-NLP/opus-mt-tc-big-en-ko \
  --output-dir outputs/opus-en-ko-transformers
```

Use `data/transformers_en_ko_eval_contaminated.jsonl` only as a stress-test set, not as the main validation set.
