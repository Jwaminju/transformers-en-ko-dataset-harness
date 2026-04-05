---
name: translation-dataset-review
description: Use when reviewing suspect EN-to-KO translation pairs or documents, deciding keep or drop, updating the blacklist, and minimizing human review for the translation dataset harness.
---

# Translation Dataset Review

Use this skill when the user asks to inspect mismatched chunks, review suspect documents, or reduce human review volume in the translation dataset pipeline.

## Read First

- [AGENTS.md](../../../AGENTS.md)
- [translation_dataset_harness.md](../../../docs/translation_dataset_harness.md)
- `data/review/train_clean_review.md`
- `data/review/eval_contaminated_review.md`
- `data/review/clean_suspect_review.md`

## Decision Standard

Prefer `drop` unless the pair is clearly usable for translation fine-tuning.

Acceptable:

- direct translation
- conservative paraphrase that preserves the same idea unit
- code-like English that belongs in technical docs

Reject:

- localized rewrite
- reordered or shifted content
- missing sentences that change meaning materially
- explanatory English left in target outside code-like spans
- section or chunk boundaries that clearly drift

## Review Outputs

Use one of:

- `keep`
- `drop`
- `needs_human_review`

Use one reason:

- `parallel_translation`
- `acceptable_paraphrase`
- `localized_rewrite`
- `partial_translation`
- `misaligned_chunk`

## Review Priorities

Review in this order:

1. first chunk of each suspect document
2. documents with large ratio variance
3. documents where headings or code-block structure diverge
4. documents already showing mixed keep/drop evidence

## Blacklist Policy

- Add a document to `data/blacklist.txt` when one or more chunks show clear structural mismatch and the document no longer looks safely salvageable.
- Do not blacklist based on minor wording differences alone.
- If uncertain, leave the document out of the blacklist and mark it for human review.

## Guardrails

- Do not rewrite the data while judging it.
- Do not merge separate review roles into one pass when a narrower judgment will do.
- Keep the reasoning short and explicit so the next reviewer can audit it quickly.
