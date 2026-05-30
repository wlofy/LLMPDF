"""Tests for the multi-source loaders."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.loaders import MarkdownLoader, PDFLoader, SQLLoader


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------


class TestPDFLoader:
    def test_raises_when_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDFLoader().load(str(tmp_path / "missing.pdf"))

    def test_raises_when_not_a_pdf(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("hi")
        with pytest.raises(ValueError, match="not a PDF"):
            PDFLoader().load(str(f))

    def test_sets_source_metadata(self, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        fake_docs = [Document(page_content="page", metadata={"page": 0})]

        with patch("src.loaders.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = fake_docs
            out = PDFLoader().load(str(pdf_file))

        assert out[0].metadata["source_type"] == "pdf"
        assert out[0].metadata["source"].endswith("doc.pdf")


# ---------------------------------------------------------------------------
# MarkdownLoader
# ---------------------------------------------------------------------------


class TestMarkdownLoader:
    def test_raises_when_path_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MarkdownLoader().load(str(tmp_path / "nope"))

    def test_loads_single_file(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# Hello\n\nbody")
        docs = MarkdownLoader().load(str(f))
        assert len(docs) == 1
        assert "Hello" in docs[0].page_content
        assert docs[0].metadata["source_type"] == "markdown"
        assert docs[0].metadata["filename"] == "note.md"

    def test_loads_directory_recursive(self, tmp_path):
        (tmp_path / "a.md").write_text("alpha")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.markdown").write_text("beta")
        (tmp_path / "c.txt").write_text("gamma")
        (tmp_path / "ignored.py").write_text("def x(): pass")

        docs = MarkdownLoader().load(str(tmp_path), recursive=True)
        contents = {d.page_content for d in docs}
        assert {"alpha", "beta", "gamma"} <= contents
        assert not any("def x" in d.page_content for d in docs)

    def test_directory_non_recursive_skips_subdirs(self, tmp_path):
        (tmp_path / "a.md").write_text("alpha")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.md").write_text("beta")

        docs = MarkdownLoader().load(str(tmp_path), recursive=False)
        contents = {d.page_content for d in docs}
        assert contents == {"alpha"}

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "real.md").write_text("real content")
        docs = MarkdownLoader().load(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].page_content == "real content"


# ---------------------------------------------------------------------------
# SQLLoader
# ---------------------------------------------------------------------------


class TestSQLLoader:
    @pytest.fixture
    def sqlite_url(self, tmp_path):
        """Build a sqlite DB and yield its connection URL."""
        import sqlite3

        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT, body TEXT)"
        )
        conn.executemany(
            "INSERT INTO articles (title, body) VALUES (?, ?)",
            [
                ("First", "alpha content"),
                ("Second", "beta content"),
                ("Third", "gamma content"),
            ],
        )
        conn.commit()
        conn.close()
        yield f"sqlite:///{db}"

    def test_loads_full_table(self, sqlite_url):
        docs = SQLLoader(sqlite_url).load(table="articles", id_column="id")
        assert len(docs) == 3
        contents = " ".join(d.page_content for d in docs)
        assert "alpha content" in contents
        assert all(d.metadata["source_type"] == "sql" for d in docs)
        assert all("row_id" in d.metadata for d in docs)

    def test_respects_row_limit(self, sqlite_url):
        docs = SQLLoader(sqlite_url).load(table="articles", row_limit=2)
        assert len(docs) == 2

    def test_custom_query(self, sqlite_url):
        docs = SQLLoader(sqlite_url).load(
            query="SELECT title FROM articles WHERE id = 2"
        )
        assert len(docs) == 1
        assert "Second" in docs[0].page_content

    def test_rejects_both_table_and_query(self, sqlite_url):
        with pytest.raises(ValueError, match="exactly one"):
            SQLLoader(sqlite_url).load(table="articles", query="SELECT 1")

    def test_rejects_neither_table_nor_query(self, sqlite_url):
        with pytest.raises(ValueError, match="exactly one"):
            SQLLoader(sqlite_url).load()
