"""LLM interface: Q&A and summarization over indexed PDF content."""

from typing import Any, Dict, List, Optional

from langchain_classic.chains import RetrievalQA
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .config import Config
from .vector_store import VectorStore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_QA_PROMPT_TEMPLATE = """You are a knowledgeable assistant that answers questions
based strictly on the provided PDF content.

Context:
{context}

Question: {question}

Provide a clear, concise answer using only information from the context above.
If the answer is not found in the context, say "I don't have enough information
in the document to answer that question."
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
    ):
        self._vector_store = vector_store
        self._llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=api_key or Config.OPENAI_API_KEY,
        )

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def answer(self, question: str, k: int = Config.SEARCH_TOP_K) -> Dict[str, Any]:
        """Answer a question using context retrieved from the vector store.

        Args:
            question: Natural-language question about the PDF content.
            k: Number of chunks to retrieve as context.

        Returns:
            Dictionary with keys:
              - ``result``: The generated answer string.
              - ``source_documents``: List of Document objects used as context.

        Raises:
            RuntimeError: If the vector store is not ready.
        """
        if not self._vector_store.is_ready:
            raise RuntimeError(
                "Vector store is not initialised. Index a PDF first."
            )

        retriever = self._vector_store.get_retriever(k=k)
        chain = RetrievalQA.from_chain_type(
            llm=self._llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": _QA_PROMPT},
        )
        return chain.invoke({"query": question})

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

        Args:
            query: Natural-language search query.
            k: Number of results to return.

        Returns:
            List of dicts, each containing:
              - ``content``: The matching text chunk.
              - ``score``: Similarity score (lower is more similar for L2).
              - ``metadata``: Source metadata (page number, file, etc.).

        Raises:
            RuntimeError: If the vector store is not ready.
        """
        if not self._vector_store.is_ready:
            raise RuntimeError(
                "Vector store is not initialised. Index a PDF first."
            )

        results = self._vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]
