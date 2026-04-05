from scripts.build_suspect_review import build_markdown


def test_build_markdown_for_suspects():
    pair_rows = [
        {
            "relative_path": "guide/example.md",
            "chunk_id": 2,
            "text": "English source sentence.",
            "target": "한국어 번역 문장입니다.",
            "source_chars": 24,
            "target_chars": 14,
        }
    ]
    suspect_rows = [
        {
            "relative_path": "guide/example.md",
            "github_url": "https://example.com",
            "max_ratio": "1.71",
            "avg_ratio": "1.71",
        }
    ]

    markdown = build_markdown(
        pair_rows,
        suspect_rows,
        max_docs=5,
        max_chunks_per_doc=1,
        truncate_chars=500,
    )

    assert "Clean Suspect Review" in markdown
    assert "guide/example.md" in markdown
    assert "ratio: 1.71" in markdown
    assert "English source sentence." in markdown
