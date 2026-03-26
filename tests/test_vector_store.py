"""Tests for VectorStore."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_docs(texts):
    return [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_embeddings():
    with patch("src.vector_store.OpenAIEmbeddings") as MockEmb:
        yield MockEmb.return_value


@pytest.fixture()
def store(mock_embeddings):
    return VectorStore(api_key="test-key")


# ---------------------------------------------------------------------------
# Tests: is_ready
# ---------------------------------------------------------------------------

class TestIsReady:
    def test_not_ready_initially(self, store):
        assert store.is_ready is False

    def test_ready_after_build(self, store):
        docs = _make_docs(["Hello world."])
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = MagicMock()
            store.build(docs)
        assert store.is_ready is True


# ---------------------------------------------------------------------------
# Tests: build
# ---------------------------------------------------------------------------

class TestBuild:
    def test_raises_on_empty_documents(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.build([])

    def test_calls_faiss_from_documents(self, store):
        docs = _make_docs(["Document one.", "Document two."])
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = MagicMock()
            store.build(docs)
        MockFAISS.from_documents.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: add_documents
# ---------------------------------------------------------------------------

class TestAddDocuments:
    def test_raises_when_not_initialised(self, store):
        with pytest.raises(RuntimeError, match="not initialised"):
            store.add_documents(_make_docs(["test"]))

    def test_delegates_to_faiss_add(self, store):
        docs = _make_docs(["First document."])
        mock_faiss = MagicMock()
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_faiss
            store.build(docs)

        new_docs = _make_docs(["Second document."])
        store.add_documents(new_docs)
        mock_faiss.add_documents.assert_called_once_with(new_docs)


# ---------------------------------------------------------------------------
# Tests: save and load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_raises_when_not_initialised(self, store, tmp_path):
        with pytest.raises(RuntimeError, match="not initialised"):
            store.save(str(tmp_path))

    def test_save_delegates_to_faiss(self, store, tmp_path):
        docs = _make_docs(["Some text."])
        mock_faiss = MagicMock()
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_faiss
            store.build(docs)
        store.save(str(tmp_path))
        mock_faiss.save_local.assert_called_once_with(str(tmp_path))

    def test_load_raises_when_directory_missing(self, store, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.load(str(tmp_path / "nonexistent"))

    def test_load_delegates_to_faiss(self, store, tmp_path):
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.load_local.return_value = MagicMock()
            store.load(str(index_dir))
        MockFAISS.load_local.assert_called_once()
        assert store.is_ready is True


# ---------------------------------------------------------------------------
# Tests: similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    def test_raises_when_not_initialised(self, store):
        with pytest.raises(RuntimeError, match="not initialised"):
            store.similarity_search("query")

    def test_delegates_to_faiss_search(self, store):
        mock_faiss = MagicMock()
        mock_faiss.similarity_search.return_value = _make_docs(["result"])
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_faiss
            store.build(_make_docs(["seed document"]))

        results = store.similarity_search("test query", k=2)
        mock_faiss.similarity_search.assert_called_once_with("test query", k=2)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests: get_retriever
# ---------------------------------------------------------------------------

class TestGetRetriever:
    def test_raises_when_not_initialised(self, store):
        with pytest.raises(RuntimeError, match="not initialised"):
            store.get_retriever()

    def test_delegates_to_faiss_as_retriever(self, store):
        mock_faiss = MagicMock()
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_faiss
            store.build(_make_docs(["text"]))

        store.get_retriever(k=3)
        mock_faiss.as_retriever.assert_called_once_with(search_kwargs={"k": 3})



class TestSimilaritySearchWithScore:
    def test_raises_when_not_initialised(self, store):
        with pytest.raises(RuntimeError, match="not initialised"):
            store.similarity_search_with_score("query")

    def test_returns_doc_score_tuples(self, store):
        doc = Document(page_content="result", metadata={})
        mock_faiss = MagicMock()
        mock_faiss.similarity_search_with_score.return_value = [(doc, 0.12)]
        with patch("src.vector_store.FAISS") as MockFAISS:
            MockFAISS.from_documents.return_value = mock_faiss
            store.build(_make_docs(["seed"]))

        results = store.similarity_search_with_score("query", k=1)
        assert results[0][0] is doc
        assert results[0][1] == pytest.approx(0.12)
