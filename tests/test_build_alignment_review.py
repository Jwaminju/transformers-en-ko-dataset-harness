from scripts.build_alignment_review import build_markdown


def test_build_markdown_contains_source_target_and_link():
    rows = [
        {
            "relative_path": "guide/example.md",
            "chunk_id": 1,
            "text": "English source sentence.",
            "target": "한국어 번역 문장입니다.",
        }
    ]

    markdown = build_markdown(
        rows,
        title="Review",
        max_docs=10,
        max_chunks_per_doc=3,
        truncate_chars=None,
    )

    assert "# Review" in markdown
    assert "guide/example.md" in markdown
    assert "English source sentence." in markdown
    assert "한국어 번역 문장입니다." in markdown
    assert "https://github.com/huggingface/transformers/blob/main/docs/source/ko/guide/example.md" in markdown
