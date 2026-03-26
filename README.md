# LLMPDF

A LLM-based PDF semantic meaning extractor that extracts, processes, and queries information from PDF documents. It comes with intelligent search and summarization powered by large language models.

## Features

- **Extract** – Load and parse multi-page PDF files
- **Semantic search** – Find relevant passages using vector embeddings (FAISS)
- **Question answering** – Ask natural-language questions and receive grounded answers with source citations
- **Summarization** – Summarize entire documents or individual pages using a map-reduce LLM chain
- **Persistent index** – Save and reload the FAISS vector index to avoid re-embedding large documents

## Project Structure

```
LLMPDF/
├── main.py                 # CLI entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration (env vars)
│   ├── pdf_processor.py    # PDF loading & chunking
│   ├── vector_store.py     # FAISS vector store
│   ├── llm_interface.py    # Q&A, summarization, search
│   └── pdf_reader.py       # High-level façade
└── tests/
    ├── test_pdf_processor.py
    ├── test_vector_store.py
    ├── test_llm_interface.py
    └── test_pdf_reader.py
```

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## CLI Usage

### Index a PDF

```bash
python main.py index document.pdf --save ./my_index
```

### Ask a question

```bash
# From a saved index:
python main.py ask "What are the main conclusions?" --load ./my_index

# Index on the fly:
python main.py ask "What are the main conclusions?" --pdf document.pdf

# Show source chunks:
python main.py ask "What is the methodology?" --pdf document.pdf --sources
```

### Summarize

```bash
# Entire document:
python main.py summarize document.pdf

# Single page (0-based):
python main.py summarize document.pdf --page 0
```

### Semantic search

```bash
python main.py search "neural networks" document.pdf --top-k 3
```

## Python API

```python
from src.pdf_reader import PDFReader

reader = PDFReader(api_key="sk-...")

# Index a PDF (returns number of chunks)
n = reader.index("document.pdf")

# Ask a question
answer = reader.ask("What is the main topic?")

# Ask with source citations
result = reader.ask_with_sources("What methodology was used?")
print(result["answer"])
for src in result["sources"]:
    print(src["metadata"], src["content"][:200])

# Summarize
summary = reader.summarize()

# Summarize a specific page
page_summary = reader.summarize_page(0)

# Semantic search
results = reader.search("deep learning", k=5)
for r in results:
    print(r["score"], r["content"][:100])

# Save / load index
reader.save_index("./my_index")
reader.load_index("./my_index")
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Configuration

All settings can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Chat model for Q&A and summarization |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `SEARCH_TOP_K` | `5` | Default number of search results |
