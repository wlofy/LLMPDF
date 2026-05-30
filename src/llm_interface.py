"""LLM interface: Q&A and summarization over indexed content.

Supports optional cross-encoder reranking: FAISS over-fetches a candidate
pool, the reranker reorders it, and the top-k context goes to the LLM.
"""

from typing import Any, Dict, List, Optional

from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .config import Config
from .reranker import CrossEncoderReranker
from .vector_store import VectorStore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QA_PROMPT_TEMPLATE = """You are a knowledgeable assistant that answers questions
based strictly on the provided context drawn from the user's knowledge base.

Context:
{context}

Question: {question}

Provide a clear, concise answer using only information from the context above.
If the answer is not found in the context, say "I don't have enough information
in the knowledge base to answer that question."
"""

_QA_PROMPT = PromptTemplate(
    template=_QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
)

_MAP_PROMPT_TEMPLATE = """Write a concise summary of the following text:
"{text}"
CONCISE SUMMARY:"""

_MAP_PROMPT = PromptTemplate(
    template=_MAP_PROMPT_TEMPLATE, input_variables=["text"]
)

_COMBINE_PROMPT_TEMPLATE = """Write a comprehensive summary of the following
summaries, capturing all key points and themes:
"{text}"
COMPREHENSIVE SUMMARY:"""

_COMBINE_PROMPT = PromptTemplate(
    template=_COMBINE_PROMPT_TEMPLATE, input_variables=["text"]
)


class LLMInterface:
    """Provides Q&A and summarization capabilities over PDF documents."""

    def __init__(
        self,
        vector_store: VectorStore,
        api_key: Optional[str] = None,
        model: str = Config.OPENAI_MODEL,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_fetch_multiplier: int = Config.RERANK_FETCH_MULTIPLIER,
    ):
        self._vector_store = vector_store
        self._reranker = reranker
        self._rerank_fetch_multiplier = max(1, rerank_fetch_multiplier)
        self._llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=api_key or Config.OPENAI_API_KEY,
        )

    # ------------------------------------------------------------------
    # Retrieval (with optional reranking)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = Config.SEARCH_TOP_K) -> List[Document]:
        """Retrieve top-k context chunks, applying the reranker if configured."""
        if not self._vector_store.is_ready:
            raise RuntimeError(
                "Vector store is not initialised. Index a source first."
            )

        if self._reranker is None:
            return self._vector_store.similarity_search(query, k=k)

        fetch_k = k * self._rerank_fetch_multiplier
        candidates = self._vector_store.similarity_search(query, k=fetch_k)
        return self._reranker.rerank(query, candidates, top_k=k)

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def answer(self, question: str, k: int = Config.SEARCH_TOP_K) -> Dict[str, Any]:
        """Answer a question using context retrieved from the vector store.

        Uses :meth:`retrieve` so reranking (when configured) is applied
        before the context is stuffed into the prompt.

        Returns:
            Dict with ``result`` (answer text) and ``source_documents``.
        """
        docs = self.retrieve(question, k=k)
        context = "\n\n".join(d.page_content for d in docs)
        prompt = _QA_PROMPT.format(context=context, question=question)
        response = self._llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response))
        return {"result": answer_text, "source_documents": docs}

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize(self, documents: List[Document]) -> str:
        """Generate a summary of the given documents.

        Uses a map-reduce strategy so that large documents exceeding the
        model's context window are handled gracefully.

        Args:
            documents: List of Document objects to summarize.

        Returns:
            Summary string.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot summarize an empty document list.")

        chain = load_summarize_chain(
            self._llm,
            chain_type="map_reduce",
            map_prompt=_MAP_PROMPT,
            combine_prompt=_COMBINE_PROMPT,
        )
        result = chain.invoke({"input_documents": documents})
        return result.get("output_text", "")

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = Config.SEARCH_TOP_K) -> List[Dict[str, Any]]:
        """Perform semantic search and return ranked results with scores.

        When a reranker is configured, results are over-fetched from FAISS
        and reordered by cross-encoder relevance. The ``score`` field is the
        cross-encoder score in that case, otherwise the FAISS L2 distance.

        Returns:
            List of dicts with ``content``, ``score``, and ``metadata``.
        """
        if not self._vector_store.is_ready:
            raise RuntimeError(
                "Vector store is not initialised. Index a source first."
            )

        if self._reranker is None:
            results = self._vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                }
                for doc, score in results
            ]

        fetch_k = k * self._rerank_fetch_multiplier
        candidates = self._vector_store.similarity_search(query, k=fetch_k)
        scored = self._reranker.rerank_with_scores(query, candidates)[:k]
        return [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
            for doc, score in scored
        ]
