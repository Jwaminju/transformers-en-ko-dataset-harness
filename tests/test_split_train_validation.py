from scripts.split_train_validation import split_by_document


def test_split_by_document_keeps_doc_groups_together():
    rows = [
        {"relative_path": "a.md", "chunk_id": 1},
        {"relative_path": "a.md", "chunk_id": 2},
        {"relative_path": "b.md", "chunk_id": 1},
        {"relative_path": "c.md", "chunk_id": 1},
    ]

    train_rows, validation_rows = split_by_document(rows, validation_ratio=0.34, seed=1)

    train_paths = {row["relative_path"] for row in train_rows}
    validation_paths = {row["relative_path"] for row in validation_rows}

    assert train_paths.isdisjoint(validation_paths)
    assert len(validation_paths) == 1
