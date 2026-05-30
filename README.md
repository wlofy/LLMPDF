# LLMPDF

A multi-source RAG knowledge base. Ingest PDFs, markdown notes, and SQL rows
into a single FAISS index, then ask questions with optional cross-encoder
reranking for higher retrieval precision.

## Features

- **Multi-source ingestion** — PDFs, markdown/text directories, and SQL tables or queries (SQLAlchemy)
- **Semantic search** — FAISS vector store backed by OpenAI embeddings
- **Cross-encoder reranking** — local `ms-marco-MiniLM` reranker (sentence-transformers) reorders FAISS candidates for higher precision; no external API needed
- **Question answering** — grounded answers with source citations
- **Summarization** — map-reduce summarization for PDFs
- **Retrieval evaluation** — `hit@k` harness compares baseline FAISS vs. reranked retrieval

## Project Structure

```
LLMPDF/
├── main.py                 # CLI entry point
├── requirements.txt
├── examples/
│   └── eval_dataset.json   # Sample retrieval eval set
├── src/
│   ├── config.py           # Env-driven configuration
│   ├── loaders.py          # PDF / Markdown / SQL loaders
│   ├── pdf_processor.py    # PDF loader + chunker (legacy)
│   ├── vector_store.py     # FAISS vector store
│   ├── reranker.py         # CrossEncoderReranker
│   ├── llm_interface.py    # Retrieval, reranking, Q&A, summarization
│   ├── knowledge_base.py   # Multi-source facade (new primary entry point)
│   ├── pdf_reader.py       # PDF-only facade (kept for backward compat)
│   └── eval.py             # Retrieval eval harness
└── tests/
```

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env  # then set OPENAI_API_KEY=sk-...
```

The first run with `--rerank` downloads the cross-encoder model (~90 MB) and
caches it locally.

## CLI Usage

### Single PDF (legacy)

```bash
python main.py index document.pdf --save ./my_index
python main.py ask "What are the main conclusions?" --load ./my_index
```

### Multi-source ingestion

```bash
python main.py ingest \
    --pdf paper.pdf \
    --md ./notes \
    --sql sqlite:///data.db --sql-table articles --sql-id-column id \
    --save ./kb
```

`--pdf` and `--md` are repeatable. `--sql` accepts a SQLAlchemy URL plus
either `--sql-table` or `--sql-query`.

### Ask a question (with reranking)

```bash
python main.py ask "What did the paper conclude about X?" \
    --load ./kb --rerank --sources
```

`--rerank` over-fetches `top_k * RERANK_FETCH_MULTIPLIER` candidates from
FAISS, scores them with the cross-encoder, and returns the best `top_k`.

### Semantic search

```bash
python main.py search "vector similarity" --load ./kb --rerank --top-k 5
```

### Retrieval evaluation

```bash
python main.py eval examples/eval_dataset.json --load ./kb --top-k 5
```

Runs the eval twice (baseline FAISS only, then with reranking) and prints
`hit@1`, `hit@3`, `hit@k` plus the delta. Dataset format:

```json
[
  {
    "question": "What is FAISS used for?",
    "keywords": ["FAISS", "vector"],
    "source_contains": "vector_store"
  }
]
```

A retrieved chunk counts as a hit if it matches all provided constraints.

## Python API

```python
from src.knowledge_base import KnowledgeBase

kb = KnowledgeBase(api_key="sk-...", use_reranker=True)

kb.add_pdf("paper.pdf")
kb.add_markdown("./notes", recursive=True)
kb.add_sql(
    "sqlite:///data.db",
    table="articles",
    id_column="id",
    row_limit=10_000,
)

print(kb.ask("What does the paper say about RAG?"))

result = kb.ask_with_sources("What methodology was used?")
print(result["answer"])
for src in result["sources"]:
    print(src["metadata"], src["content"][:200])

results = kb.search("vector similarity", k=5)
for r in results:
    print(r["score"], r["content"][:100])

kb.save_index("./kb")
kb.load_index("./kb")
```

The legacy single-PDF `PDFReader` is still available for backward
compatibility — see `src/pdf_reader.py`.

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Chat model for Q&A |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SEARCH_TOP_K` | `5` | Default top-k for retrieval |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RERANK_FETCH_MULTIPLIER` | `5` | Candidates fetched = top_k × this |
