"""Tests for LLMInterface."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.llm_interface import LLMInterface
from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(texts):
    return [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(texts)]


def _make_ready_vector_store():
    """Return a VectorStore whose is_ready property returns True."""
    vs = MagicMock(spec=VectorStore)
    vs.is_ready = True
    return vs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vector_store():
    return _make_ready_vector_store()


@pytest.fixture()
def llm_interface(vector_store):
    with patch("src.llm_interface.ChatOpenAI"):
        return LLMInterface(vector_store=vector_store, api_key="test-key")


# ---------------------------------------------------------------------------
# Tests: answer
# ---------------------------------------------------------------------------

class TestAnswer:
    def test_raises_when_vector_store_not_ready(self):
        vs = MagicMock(spec=VectorStore)
        vs.is_ready = False
        with patch("src.llm_interface.ChatOpenAI"):
            iface = LLMInterface(vector_store=vs, api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialised"):
            iface.answer("What is this?")

    def test_returns_result_string(self, llm_interface, vector_store):
        vector_store.similarity_search.return_value = _make_docs(["Context chunk."])
        llm_interface._llm = MagicMock()
        llm_interface._llm.invoke.return_value = MagicMock(content="Test answer.")

        result = llm_interface.answer("What is this document about?")

        assert result["result"] == "Test answer."
        assert len(result["source_documents"]) == 1

    def test_retriever_uses_correct_k(self, llm_interface, vector_store):
        vector_store.similarity_search.return_value = []
        llm_interface._llm = MagicMock()
        llm_interface._llm.invoke.return_value = MagicMock(content="ok")

        llm_interface.answer("question", k=7)

        vector_store.similarity_search.assert_called_once_with("question", k=7)

    def test_reranker_overfetches_and_reorders(self, vector_store):
        reranker = MagicMock()
        candidates = _make_docs(["a", "b", "c", "d", "e"])
        vector_store.similarity_search.return_value = candidates
        reranker.rerank.return_value = [candidates[3], candidates[0]]

        with patch("src.llm_interface.ChatOpenAI"):
            iface = LLMInterface(
                vector_store=vector_store,
                api_key="k",
                reranker=reranker,
                rerank_fetch_multiplier=3,
            )
        iface._llm = MagicMock()
        iface._llm.invoke.return_value = MagicMock(content="ok")

        result = iface.answer("q", k=2)

        # Over-fetched k * multiplier = 6 candidates from FAISS.
        vector_store.similarity_search.assert_called_once_with("q", k=6)
        # Reranker received those candidates, top_k=2.
        reranker.rerank.assert_called_once_with("q", candidates, top_k=2)
        assert result["source_documents"] == [candidates[3], candidates[0]]


# ---------------------------------------------------------------------------
# Tests: summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def test_raises_on_empty_documents(self, llm_interface):
        with pytest.raises(ValueError, match="empty"):
            llm_interface.summarize([])

    def test_returns_summary_text(self, llm_interface):
        docs = _make_docs(["This is the content of the document."])
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output_text": "Summary of document."}

        with patch("src.llm_interface.load_summarize_chain") as mock_load:
            mock_load.return_value = mock_chain
            summary = llm_interface.summarize(docs)

        assert summary == "Summary of document."

    def test_uses_map_reduce_chain_type(self, llm_interface):
        docs = _make_docs(["content"])
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"output_text": "ok"}

        with patch("src.llm_interface.load_summarize_chain") as mock_load:
            mock_load.return_value = mock_chain
            llm_interface.summarize(docs)

        _, kwargs = mock_load.call_args
        assert kwargs.get("chain_type") == "map_reduce"


# ---------------------------------------------------------------------------
# Tests: search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_raises_when_vector_store_not_ready(self):
        vs = MagicMock(spec=VectorStore)
        vs.is_ready = False
        with patch("src.llm_interface.ChatOpenAI"):
            iface = LLMInterface(vector_store=vs, api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialised"):
            iface.search("query")

    def test_returns_formatted_results(self, llm_interface, vector_store):
        doc = Document(
            page_content="Relevant passage.",
            metadata={"page": 2, "source": "doc.pdf"},
        )
        vector_store.similarity_search_with_score.return_value = [(doc, 0.25)]

        results = llm_interface.search("some query", k=1)

        assert len(results) == 1
        assert results[0]["content"] == "Relevant passage."
        assert results[0]["score"] == pytest.approx(0.25)
        assert results[0]["metadata"]["page"] == 2

    def test_passes_k_to_vector_store(self, llm_interface, vector_store):
        vector_store.similarity_search_with_score.return_value = []
        llm_interface.search("query", k=4)
        vector_store.similarity_search_with_score.assert_called_once_with(
            "query", k=4
        )
