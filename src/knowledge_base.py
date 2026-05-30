"""KnowledgeBase: multi-source RAG facade over PDFs, Markdown notes, and SQL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Config
from .llm_interface import LLMInterface
from .loaders import MarkdownLoader, PDFLoader, SQLLoader
from .reranker import CrossEncoderReranker
from .vector_store import VectorStore


class KnowledgeBase:
    """End-to-end multi-source RAG.

    Ingest PDFs, Markdown/text notes, and SQL rows into a single FAISS
    index, then ask questions with optional cross-encoder reranking::

        kb = KnowledgeBase(api_key="sk-...", use_reranker=True)
        kb.add_pdf("paper.pdf")
        kb.add_markdown("./notes", recursive=True)
        kb.add_sql("sqlite:///data.db", table="articles", id_column="id")
        print(kb.ask("what does the paper say about X?"))
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = Config.OPENAI_MODEL,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
        use_reranker: bool = False,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self._vector_store = VectorStore(api_key=api_key)

        if reranker is not None:
            active_reranker: Optional[CrossEncoderReranker] = reranker
        elif use_reranker:
            active_reranker = CrossEncoderReranker()
        else:
            active_reranker = None

        self._llm = LLMInterface(
            vector_store=self._vector_store,
            api_key=api_key,
            model=model,
            reranker=active_reranker,
        )
        self._documents: List[Document] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_pdf(self, pdf_path: str) -> int:
        """Ingest a PDF file. Returns the number of chunks added."""
        docs = PDFLoader().load(pdf_path)
        return self._ingest(docs)

    def add_markdown(
        self,
        path: str,
        recursive: bool = True,
        globs: Optional[Sequence[str]] = None,
    ) -> int:
        """Ingest a markdown/text file or directory. Returns chunks added."""
        loader = MarkdownLoader(globs=globs) if globs else MarkdownLoader()
        docs = loader.load(path, recursive=recursive)
        return self._ingest(docs)

    def add_sql(
        self,
        connection_url: str,
        table: Optional[str] = None,
        query: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        id_column: Optional[str] = None,
        row_limit: Optional[int] = None,
    ) -> int:
        """Ingest rows from a SQL database. Returns chunks added."""
        docs = SQLLoader(connection_url).load(
            table=table,
            query=query,
            columns=columns,
            id_column=id_column,
            row_limit=row_limit,
        )
        return self._ingest(docs)

    def _ingest(self, docs: List[Document]) -> int:
        if not docs:
            return 0
        self._documents.extend(docs)
        chunks = self._splitter.split_documents(docs)
        if not self._vector_store.is_ready:
            self._vector_store.build(chunks)
        else:
            self._vector_store.add_documents(chunks)
        return len(chunks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, directory: str) -> None:
        self._vector_store.save(directory)

    def load_index(self, directory: str) -> None:
        self._vector_store.load(directory)

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def ask(self, question: str, k: int = Config.SEARCH_TOP_K) -> str:
        return self._llm.answer(question, k=k).get("result", "")

    def ask_with_sources(
        self, question: str, k: int = Config.SEARCH_TOP_K
    ) -> Dict[str, Any]:
        result = self._llm.answer(question, k=k)
        sources = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in result.get("source_documents", [])
        ]
        return {"answer": result.get("result", ""), "sources": sources}

    def search(
        self, query: str, k: int = Config.SEARCH_TOP_K
    ) -> List[Dict[str, Any]]:
        return self._llm.search(query, k=k)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def has_reranker(self) -> bool:
        return self._llm._reranker is not None  # noqa: SLF001 (intentional)

    @property
    def document_count(self) -> int:
        return len(self._documents)
