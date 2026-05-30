"""Source loaders: produce LangChain Documents from PDFs, Markdown notes, and SQL."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseLoader:
    """Common interface for source loaders.

    Subclasses implement :meth:`load` and return one or more
    :class:`Document` objects with at least a ``source`` metadata key so
    retrieved chunks remain traceable to their origin.
    """

    source_type: str = "unknown"

    def load(self, *args, **kwargs) -> List[Document]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class PDFLoader(BaseLoader):
    """Load a PDF as one Document per page."""

    source_type = "pdf"

    def load(self, pdf_path: str) -> List[Document]:
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        docs = PyPDFLoader(str(path)).load()
        for d in docs:
            d.metadata.setdefault("source", str(path))
            d.metadata["source_type"] = self.source_type
        return docs


# ---------------------------------------------------------------------------
# Markdown / plain-text notes
# ---------------------------------------------------------------------------


class MarkdownLoader(BaseLoader):
    """Load markdown/plain-text notes from a single file or a directory."""

    source_type = "markdown"

    DEFAULT_GLOBS: Sequence[str] = ("*.md", "*.markdown", "*.txt")

    def __init__(self, globs: Optional[Sequence[str]] = None):
        self._globs = tuple(globs) if globs else self.DEFAULT_GLOBS

    def load(self, path: str, recursive: bool = True) -> List[Document]:
        """Load notes from a file or directory.

        Args:
            path: File or directory containing notes.
            recursive: When ``path`` is a directory, descend into subfolders.

        Returns:
            One Document per file. Splitting is the caller's responsibility.
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Notes path not found: {path}")

        files: List[Path]
        if p.is_file():
            files = [p]
        else:
            files = sorted(self._iter_files(p, recursive))

        docs: List[Document] = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(f),
                        "source_type": self.source_type,
                        "filename": f.name,
                    },
                )
            )
        return docs

    def _iter_files(self, root: Path, recursive: bool) -> Iterable[Path]:
        for pattern in self._globs:
            it = root.rglob(pattern) if recursive else root.glob(pattern)
            for f in it:
                if f.is_file():
                    yield f


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


class SQLLoader(BaseLoader):
    """Embed rows from a SQL database as Documents.

    Each row is serialised as ``col1: value1\\ncol2: value2\\n...`` so the
    embedding model sees both column names and values. Use either
    ``table`` (full table dump) or ``query`` (arbitrary SELECT).
    """

    source_type = "sql"

    def __init__(self, connection_url: str):
        # Imported lazily so the dependency is optional for PDF-only users.
        from sqlalchemy import create_engine

        self._engine = create_engine(connection_url)
        self._connection_url = connection_url

    def load(
        self,
        table: Optional[str] = None,
        query: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        id_column: Optional[str] = None,
        row_limit: Optional[int] = None,
    ) -> List[Document]:
        """Load rows as Documents.

        Args:
            table: Table name to dump (mutually exclusive with ``query``).
            query: Raw SELECT to execute (mutually exclusive with ``table``).
            columns: Optional column allow-list for table mode.
            id_column: Column to record as ``row_id`` in metadata.
            row_limit: Maximum number of rows to load.
        """
        if bool(table) == bool(query):
            raise ValueError("Provide exactly one of `table` or `query`.")

        from sqlalchemy import text

        if table:
            col_clause = ", ".join(columns) if columns else "*"
            sql = f"SELECT {col_clause} FROM {table}"
            if row_limit:
                sql += f" LIMIT {int(row_limit)}"
            source_label = f"sql://{table}"
        else:
            sql = query  # type: ignore[assignment]
            source_label = "sql://query"

        with self._engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.mappings().all()
            if row_limit and query:
                rows = rows[: int(row_limit)]

        docs: List[Document] = []
        for row in rows:
            content = "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)
            if not content.strip():
                continue
            meta = {
                "source": source_label,
                "source_type": self.source_type,
            }
            if id_column and id_column in row:
                meta["row_id"] = row[id_column]
            docs.append(Document(page_content=content, metadata=meta))
        return docs
