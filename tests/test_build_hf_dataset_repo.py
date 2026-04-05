from scripts.build_hf_dataset_repo import build_readme


def test_build_readme_contains_counts_and_paths():
    readme = build_readme(
        train_rows=81,
        eval_rows=880,
        validation_rows=9,
        blacklist_rows=13,
        source_commit="abc123",
        dataset_name="Demo Dataset",
    )

    assert "# Demo Dataset" in readme
    assert "train rows: 81" in readme
    assert "validation rows: 9" in readme
    assert "stress-test rows: 880" in readme
    assert "blacklist docs: 13" in readme
    assert "`data/train.jsonl`" in readme
    assert "`metadata/blacklist.txt`" in readme
    assert "`docs/translation_dataset_harness.md`" in readme
