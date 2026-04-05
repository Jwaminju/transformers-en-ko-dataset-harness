from pathlib import Path

from scripts.build_translation_corpus import build_parallel_records, strip_markdown


def test_strip_markdown_removes_frontmatter_links_and_code():
    text = """---
title: Example
---
*This model was released on 2024-03-27 and added to Hugging Face Transformers on 2024-04-18.*

# Heading

<--copyright-->
<hfoption id="demo">
[[toc]]

Paragraph with [link](https://example.com) and `code`.

```python
print("skip")
```
"""
    assert strip_markdown(text) == "Paragraph with link and code."


def test_build_parallel_records_matches_relative_paths(tmp_path: Path):
    docs_root = tmp_path / "docs" / "source"
    (docs_root / "en" / "guide").mkdir(parents=True)
    (docs_root / "ko" / "guide").mkdir(parents=True)

    (docs_root / "en" / "guide" / "intro.md").write_text(
        "# Intro\n\nThis English document is long enough to keep for training.",
        encoding="utf-8",
    )
    (docs_root / "ko" / "guide" / "intro.md").write_text(
        "# 소개\n\n이 한국어 문서는 학습에 사용할 만큼 충분히 깁니다.",
        encoding="utf-8",
    )
    (docs_root / "en" / "guide" / "english-only.md").write_text(
        "Only in English but still quite long for the filter threshold.",
        encoding="utf-8",
    )

    records, source_only, target_only = build_parallel_records(docs_root, min_chars=10)

    assert [record.relative_path for record in records] == ["guide/intro.md"]
    assert source_only == ["guide/english-only.md"]
    assert target_only == []
