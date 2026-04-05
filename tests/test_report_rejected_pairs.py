from scripts.report_rejected_pairs import build_report_rows


def test_build_report_rows_groups_by_document():
    rows = [
        {
            "relative_path": "guide/a.md",
            "github_url": "https://example.com/a",
            "reason": "high_english_ratio",
            "target": "Then launch your training with extra arguments if needed.",
        },
        {
            "relative_path": "guide/a.md",
            "github_url": "https://example.com/a",
            "reason": "length_ratio",
            "target": "짧음",
        },
        {
            "relative_path": "guide/b.md",
            "github_url": "https://example.com/b",
            "reason": "mostly_english_target",
            "target": "For more details see the documentation page.",
        },
    ]

    report_rows = build_report_rows(rows)

    assert report_rows[0].relative_path == "guide/a.md"
    assert report_rows[0].rejected_pairs == 2
    assert report_rows[0].high_english_ratio == 1
    assert report_rows[0].length_ratio == 1
    assert "Then launch" in report_rows[0].sample_target
