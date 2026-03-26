"""Tests for PDFReader (high-level façade)."""

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from src.pdf_reader import PDFReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(texts):
    return [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reader():
    with (
        patch("src.vector_store.OpenAIEmbeddings"),
        patch("src.llm_interface.ChatOpenAI"),
    ):
        return PDFReader(api_key="test-key")


# ---------------------------------------------------------------------------
# Tests: index
# ---------------------------------------------------------------------------

class TestIndex:
    def test_index_returns_chunk_count(self, reader):
        docs = _make_docs(["word " * 50])
        chunks = _make_docs(["chunk1", "chunk2", "chunk3"])

        with (
            patch.object(reader._processor, "load", return_value=docs),
            patch.object(reader._processor, "split", return_value=chunks),
            patch.object(reader._vector_store, "build") as mock_build,
        ):
            n = reader.index("fake.pdf")

        assert n == 3
        mock_build.assert_called_once_with(chunks)

    def test_index_stores_documents(self, reader):
        docs = _make_docs(["page one"])
        with (
            patch.object(reader._processor, "load", return_value=docs),
            patch.object(reader._processor, "split", return_value=docs),
            patch.object(reader._vector_store, "build"),
        ):
            reader.index("fake.pdf")

        assert reader._documents == docs

    def test_page_count_after_index(self, reader):
        docs = _make_docs(["p1", "p2", "p3"])
        with (
            patch.object(reader._processor, "load", return_value=docs),
            patch.object(reader._processor, "split", return_value=docs),
            patch.object(reader._vector_store, "build"),
        ):
            reader.index("fake.pdf")

        assert reader.page_count == 3


# ---------------------------------------------------------------------------
# Tests: ask
# ---------------------------------------------------------------------------

class TestAsk:
    def test_ask_returns_answer_string(self, reader):
        with patch.object(
            reader._llm,
            "answer",
            return_value={"result": "42.", "source_documents": []},
        ):
            answer = reader.ask("What is the answer?")

        assert answer == "42."

    def test_ask_passes_k_parameter(self, reader):
        with patch.object(
            reader._llm,
            "answer",
            return_value={"result": "ok", "source_documents": []},
        ) as mock_answer:
            reader.ask("question", k=3)

        mock_answer.assert_called_once_with("question", k=3)

    def test_ask_with_sources_returns_dict(self, reader):
        src_doc = Document(
            page_content="relevant text",
            metadata={"page": 1, "source": "doc.pdf"},
        )
        with patch.object(
            reader._llm,
            "answer",
            return_value={"result": "answer", "source_documents": [src_doc]},
        ):
            result = reader.ask_with_sources("question")

        assert result["answer"] == "answer"
        assert len(result["sources"]) == 1
        assert result["sources"][0]["content"] == "relevant text"


# ---------------------------------------------------------------------------
# Tests: summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_summarize_raises_when_no_docs(self, reader):
        with pytest.raises(RuntimeError, match="No PDF"):
            reader.summarize()

    def test_summarize_delegates_to_llm(self, reader):
        docs = _make_docs(["content"])
        reader._documents = docs

        with patch.object(reader._llm, "summarize", return_value="Summary.") as mock_sum:
            summary = reader.summarize()

        mock_sum.assert_called_once_with(docs)
        assert summary == "Summary."

    def test_summarize_page_raises_out_of_range(self, reader):
        reader._documents = _make_docs(["only page"])
        with pytest.raises(IndexError):
            reader.summarize_page(5)

    def test_summarize_page_passes_single_document(self, reader):
        docs = _make_docs(["page0", "page1", "page2"])
        reader._documents = docs

        with patch.object(reader._llm, "summarize", return_value="Page summary.") as mock_sum:
            result = reader.summarize_page(1)

        mock_sum.assert_called_once_with([docs[1]])
        assert result == "Page summary."


# ---------------------------------------------------------------------------
# Tests: search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_delegates_to_llm(self, reader):
        expected = [{"content": "text", "score": 0.1, "metadata": {}}]
        with patch.object(reader._llm, "search", return_value=expected) as mock_search:
            results = reader.search("neural networks", k=2)

        mock_search.assert_called_once_with("neural networks", k=2)
        assert results == expected


# ---------------------------------------------------------------------------
# Tests: full_text
# ---------------------------------------------------------------------------

class TestFullText:
    def test_full_text_concatenates_pages(self, reader):
        reader._documents = _make_docs(["Page A content.", "Page B content."])
        text = reader.full_text
        assert "Page A content." in text
        assert "Page B content." in text


# ---------------------------------------------------------------------------
# Tests: save/load index
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_index_delegates(self, reader):
        with patch.object(reader._vector_store, "save") as mock_save:
            reader.save_index("/tmp/idx")
        mock_save.assert_called_once_with("/tmp/idx")

    def test_load_index_delegates(self, reader):
        with patch.object(reader._vector_store, "load") as mock_load:
            reader.load_index("/tmp/idx")
        mock_load.assert_called_once_with("/tmp/idx")
