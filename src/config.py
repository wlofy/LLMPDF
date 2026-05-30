"""Configuration settings for LLMPDF."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"
    )

    # Text splitting settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Summary settings
    SUMMARY_MAX_TOKENS: int = int(os.getenv("SUMMARY_MAX_TOKENS", "500"))
    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "5"))

    # Reranker settings
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # How many candidates to over-fetch from FAISS before reranking.
    # Final pool size is SEARCH_TOP_K * RERANK_FETCH_MULTIPLIER.
    RERANK_FETCH_MULTIPLIER: int = int(os.getenv("RERANK_FETCH_MULTIPLIER", "5"))
