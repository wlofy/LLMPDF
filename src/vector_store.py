"""Vector store: builds and queries a FAISS index for semantic search."""

import os
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Config


class VectorStore:
    """Manages a FAISS vector store backed by OpenAI embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or Config.OPENAI_API_KEY
        self._embeddings = OpenAIEmbeddings(
            model=Config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self._api_key,
        )
        self._store: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, documents: List[Document]) -> None:
        """Create a new vector store from a list of documents.

        Args:
            documents: Chunked Document objects to index.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot build vector store from an empty document list.")
        self._store = FAISS.from_documents(documents, self._embeddings)

    def add_documents(self, documents: List[Document]) -> None:
        """Add more documents to an existing vector store.

        Args:
            documents: Documents to add.

        Raises:
            RuntimeError: If the vector store has not been built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. Call build() first."
            )
        self._store.add_documents(documents)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Persist the vector store to disk.

        Args:
            directory: Directory path where the index will be saved.

        Raises:
            RuntimeError: If the vector store has not been built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. Call build() first."
            )
        os.makedirs(directory, exist_ok=True)
        self._store.save_local(directory)

    def load(self, directory: str) -> None:
        """Load a previously saved vector store from disk.

        Args:
            directory: Directory path containing the saved index.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Vector store directory not found: {directory}"
            )
        self._store = FAISS.load_local(
            directory,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = Config.SEARCH_TOP_K
    ) -> List[Document]:
        """Return the *k* most semantically similar document chunks.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of the top-k matching Document objects.

        Raises:
            RuntimeError: If the vector store has not been built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. Call build() first."
            )
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = Config.SEARCH_TOP_K
    ) -> List[Tuple[Document, float]]:
        """Return the top-k chunks together with their relevance scores.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples ordered by relevance.

        Raises:
            RuntimeError: If the vector store has not been built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. Call build() first."
            )
        return self._store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = Config.SEARCH_TOP_K):
        """Return a LangChain retriever backed by this FAISS store.

        Args:
            k: Number of documents to retrieve per query.

        Returns:
            A VectorStoreRetriever.

        Raises:
            RuntimeError: If the vector store has not been built yet.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector store is not initialised. Call build() first."
            )
        return self._store.as_retriever(search_kwargs={"k": k})

    @property
    def is_ready(self) -> bool:
        """True if the vector store has been built or loaded."""
        return self._store is not None
