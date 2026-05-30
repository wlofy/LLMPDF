"""Cross-encoder reranker for boosting retrieval precision after FAISS recall."""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Rerank candidate documents with a sentence-transformers cross-encoder.

    The model is downloaded on first use (~90 MB for ms-marco-MiniLM-L-6-v2)
    and cached locally by HuggingFace. Reranking is CPU-friendly: a couple
    of hundred milliseconds for 25 candidates.

    Usage::

        reranker = CrossEncoderReranker()
        top = reranker.rerank(query, candidates, top_k=5)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # lazy

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so the dependency is optional.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def score(self, query: str, documents: List[Document]) -> List[float]:
        """Return a relevance score for each (query, document) pair."""
        if not documents:
            return []
        model = self._ensure_model()
        pairs = [(query, d.page_content) for d in documents]
        scores = model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Return ``documents`` reordered by cross-encoder relevance.

        Args:
            query: The user's natural-language query.
            documents: Candidate documents (typically from a FAISS recall pass).
            top_k: If set, truncate to the best ``top_k`` after reranking.
        """
        if not documents:
            return []
        scored = self.rerank_with_scores(query, documents)
        if top_k is not None:
            scored = scored[:top_k]
        return [d for d, _ in scored]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        """Like :meth:`rerank` but returns ``(document, score)`` pairs."""
        scores = self.score(query, documents)
        paired = list(zip(documents, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired
