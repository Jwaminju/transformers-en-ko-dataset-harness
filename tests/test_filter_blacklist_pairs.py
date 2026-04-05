from scripts.filter_blacklist_pairs import read_blacklist


def test_read_blacklist_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "blacklist.txt"
    path.write_text("# comment\n\nfoo.md\nbar/baz.md\n", encoding="utf-8")
    assert read_blacklist(path) == {"foo.md", "bar/baz.md"}
