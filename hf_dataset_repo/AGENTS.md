# Dataset Repo Entry Point

Read these first:

- `README.md`
- `docs/translation_dataset_harness.md`
- `docs/dataset_methodology.md`

Rules:

- Treat `data/` as the published artifact surface.
- Treat `metadata/` as the explanation layer for filtering and rejection decisions.
- `stress_test_contaminated.jsonl` is not the main validation set.
- If you rebuild this dataset elsewhere, keep review artifacts and blacklist decisions versioned.
