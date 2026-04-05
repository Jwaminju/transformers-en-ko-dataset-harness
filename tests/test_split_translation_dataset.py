from scripts.split_translation_dataset import split_pairs


def test_split_pairs_uses_only_clean_docs_for_training():
    pair_rows = [
        {"relative_path": "guide/clean.md", "text": "a", "target": "가"},
        {"relative_path": "guide/dirty.md", "text": "b", "target": "나"},
    ]
    rejected_rows = [
        {"relative_path": "guide/dirty.md", "reason": "high_english_ratio"},
    ]

    train_rows, eval_rows, doc_rows = split_pairs(pair_rows, rejected_rows)

    assert [row["relative_path"] for row in train_rows] == ["guide/clean.md"]
    assert [row["relative_path"] for row in eval_rows] == ["guide/dirty.md"]
    assert doc_rows == [
        {
            "relative_path": "guide/clean.md",
            "status": "clean",
            "github_url": "https://github.com/huggingface/transformers/blob/main/docs/source/ko/guide/clean.md",
        },
        {
            "relative_path": "guide/dirty.md",
            "status": "contaminated",
            "github_url": "https://github.com/huggingface/transformers/blob/main/docs/source/ko/guide/dirty.md",
        },
    ]
