"""PDF processor: loads, extracts text, and splits PDF documents into chunks."""

from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import Config


class PDFProcessor:
    """Handles loading and chunking of PDF documents."""

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load(self, pdf_path: str) -> List[Document]:
        """Load a PDF file and return its pages as Documents.

        Args:
            pdf_path: Absolute or relative path to the PDF file.

        Returns:
            List of Document objects, one per page.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the path does not point to a PDF file.
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        loader = PyPDFLoader(str(path))
        return loader.load()

    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks suitable for embedding.

        Args:
            documents: List of Document objects to split.

        Returns:
            List of chunked Document objects with preserved metadata.
        """
        return self._splitter.split_documents(documents)

    def load_and_split(self, pdf_path: str) -> List[Document]:
        """Convenience method: load a PDF and split it into chunks.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of chunked Document objects.
        """
        documents = self.load(pdf_path)
        return self.split(documents)

    @staticmethod
    def extract_text(documents: List[Document]) -> str:
        """Concatenate the page content of all documents into a single string.

        Args:
            documents: List of Document objects.

        Returns:
            Full text content as a single string.
        """
        return "\n\n".join(doc.page_content for doc in documents)
