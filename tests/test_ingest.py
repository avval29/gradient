from gradient.ingest import load_documents


def test_loads_txt_and_md_and_skips_readme(tmp_path):
    (tmp_path / "ch1.txt").write_text("plants make food by photosynthesis " * 5)
    (tmp_path / "ch2.md").write_text("# Heading\nthe heart pumps blood " * 5)
    (tmp_path / "README.md").write_text("not curriculum")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    sources = {c.source for c in load_documents(tmp_path)}
    assert sources == {"ch1.txt", "ch2.md"}


def test_empty_file_is_ignored(tmp_path):
    (tmp_path / "empty.txt").write_text("   ")
    assert load_documents(tmp_path) == []
