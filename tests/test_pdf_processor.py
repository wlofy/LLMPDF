"""Tests for PDFProcessor."""

import io
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.pdf_processor import PDFProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_documents(texts):
    """Return a list of Documents with the given page_content strings."""
    return [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def processor():
    return PDFProcessor(chunk_size=100, chunk_overlap=10)


# ---------------------------------------------------------------------------
# Tests: load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_raises_when_file_not_found(self, processor, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            processor.load(str(tmp_path / "nonexistent.pdf"))

    def test_raises_when_not_a_pdf(self, processor, tmp_path):
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="not a PDF"):
            processor.load(str(txt_file))

    def test_load_calls_pypdf_loader(self, processor, tmp_path):
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        fake_docs = _make_documents(["Page one content.", "Page two content."])

        with patch("src.pdf_processor.PyPDFLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.load.return_value = fake_docs
            result = processor.load(str(pdf_file))

        MockLoader.assert_called_once_with(str(pdf_file))
        instance.load.assert_called_once()
        assert result == fake_docs


# ---------------------------------------------------------------------------
# Tests: split
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_produces_smaller_chunks(self, processor):
        # A document longer than chunk_size=100 should be split into more chunks.
        long_text = "word " * 100  # ~500 chars
        docs = _make_documents([long_text])
        chunks = processor.split(docs)
        assert len(chunks) > 1

    def test_split_preserves_metadata(self, processor):
        docs = _make_documents(["Short text that fits in one chunk."])
        chunks = processor.split(docs)
        assert all("page" in c.metadata for c in chunks)

    def test_split_empty_list_returns_empty(self, processor):
        assert processor.split([]) == []


# ---------------------------------------------------------------------------
# Tests: extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_concatenates_content(self):
        docs = _make_documents(["Hello", "World"])
        text = PDFProcessor.extract_text(docs)
        assert "Hello" in text
        assert "World" in text

    def test_separator_between_pages(self):
        docs = _make_documents(["Page A", "Page B"])
        text = PDFProcessor.extract_text(docs)
        assert "\n\n" in text

    def test_empty_list_returns_empty_string(self):
        assert PDFProcessor.extract_text([]) == ""


# ---------------------------------------------------------------------------
# Tests: load_and_split (integration of load + split)
# ---------------------------------------------------------------------------

class TestLoadAndSplit:
    def test_returns_chunks(self, processor, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        long_text = "word " * 200  # well over chunk_size=100
        fake_docs = _make_documents([long_text])

        with patch("src.pdf_processor.PyPDFLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.load.return_value = fake_docs
            chunks = processor.load_and_split(str(pdf_file))

        assert len(chunks) > 1
