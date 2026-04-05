from scripts.build_clean_suspects import build_suspect_rows


def test_build_suspect_rows_sorts_by_max_ratio():
    rows = [
        {
            "relative_path": "guide/a.md",
            "chunk_id": 1,
            "text": "English source sentence",
            "target": "한국어 문장",
            "source_chars": 24,
            "target_chars": 12,
        },
        {
            "relative_path": "guide/a.md",
            "chunk_id": 2,
            "text": "Another English source sentence",
            "target": "또 다른 한국어 문장입니다.",
            "source_chars": 31,
            "target_chars": 20,
        },
        {
            "relative_path": "guide/b.md",
            "chunk_id": 1,
            "text": "Short",
            "target": "충분히 긴 한국어 문장입니다.",
            "source_chars": 5,
            "target_chars": 16,
        },
    ]

    suspect_rows = build_suspect_rows(rows)

    assert suspect_rows[0]["relative_path"] == "guide/a.md"
    assert suspect_rows[0]["max_ratio"] == "2.00"
    assert suspect_rows[0]["worst_chunk_id"] == "1"
    assert suspect_rows[1]["relative_path"] == "guide/b.md"
