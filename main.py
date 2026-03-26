#!/usr/bin/env python3
"""LLMPDF – Command-line interface for the LLM-based PDF reader.

Usage examples
--------------
Index a PDF and ask a question:
    python main.py index document.pdf --save ./my_index
    python main.py ask "What are the main conclusions?" --load ./my_index

Summarize an entire document:
    python main.py summarize document.pdf

Summarize a single page (0-based):
    python main.py summarize document.pdf --page 0

Perform semantic search:
    python main.py search "machine learning" document.pdf --top-k 3
"""

import argparse
import sys

from src.config import Config
from src.pdf_reader import PDFReader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmpdf",
        description="LLM-based PDF reader – extract, search, and summarize PDF documents.",
    )
    parser.add_argument(
        "--api-key",
        default=Config.OPENAI_API_KEY or None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default=Config.OPENAI_MODEL,
        help="OpenAI chat model to use (default: %(default)s).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ index
    p_index = sub.add_parser("index", help="Index a PDF file for later querying.")
    p_index.add_argument("pdf", help="Path to the PDF file.")
    p_index.add_argument(
        "--save",
        metavar="DIR",
        help="Directory where the vector index will be saved.",
    )

    # ------------------------------------------------------------------- ask
    p_ask = sub.add_parser("ask", help="Ask a question about an indexed PDF.")
    p_ask.add_argument("question", help="Question to ask.")
    p_ask.add_argument("--pdf", help="PDF file to index before asking.")
    p_ask.add_argument(
        "--load",
        metavar="DIR",
        help="Load a previously saved vector index from this directory.",
    )
    p_ask.add_argument(
        "--sources",
        action="store_true",
        help="Show the source document chunks used to generate the answer.",
    )
    p_ask.add_argument(
        "--top-k",
        type=int,
        default=Config.SEARCH_TOP_K,
        help="Number of context chunks to retrieve (default: %(default)s).",
    )

    # --------------------------------------------------------------- summarize
    p_sum = sub.add_parser("summarize", help="Summarize a PDF document.")
    p_sum.add_argument("pdf", help="Path to the PDF file.")
    p_sum.add_argument(
        "--page",
        type=int,
        default=None,
        metavar="N",
        help="Summarize only page N (0-based index).",
    )

    # ----------------------------------------------------------------- search
    p_search = sub.add_parser(
        "search", help="Perform semantic search over a PDF document."
    )
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("pdf", help="Path to the PDF file.")
    p_search.add_argument(
        "--top-k",
        type=int,
        default=Config.SEARCH_TOP_K,
        help="Number of results to return (default: %(default)s).",
    )
    p_search.add_argument(
        "--load",
        metavar="DIR",
        help="Load a previously saved vector index from this directory.",
    )

    return parser


def _require_api_key(api_key: str) -> None:
    if not api_key:
        print(
            "Error: OpenAI API key is required. "
            "Set OPENAI_API_KEY in your .env file or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_index(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    reader = PDFReader(api_key=args.api_key, model=args.model)
    print(f"Indexing {args.pdf} …")
    n = reader.index(args.pdf)
    print(f"✓ Indexed {n} chunks from {args.pdf}.")
    if args.save:
        reader.save_index(args.save)
        print(f"✓ Index saved to {args.save}.")


def cmd_ask(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    reader = PDFReader(api_key=args.api_key, model=args.model)

    if args.load:
        print(f"Loading index from {args.load} …")
        reader.load_index(args.load)
    elif args.pdf:
        print(f"Indexing {args.pdf} …")
        reader.index(args.pdf)
    else:
        print(
            "Error: provide either --pdf to index a document or "
            "--load to load an existing index.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.sources:
        result = reader.ask_with_sources(args.question, k=args.top_k)
        print(f"\nAnswer:\n{result['answer']}\n")
        print("Sources:")
        for i, src in enumerate(result["sources"], 1):
            meta = src["metadata"]
            page = meta.get("page", "?")
            source = meta.get("source", "")
            print(f"  [{i}] Page {page} — {source}")
            print(f"      {src['content'][:200].strip()} …")
    else:
        answer = reader.ask(args.question, k=args.top_k)
        print(f"\nAnswer:\n{answer}")


def cmd_summarize(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    reader = PDFReader(api_key=args.api_key, model=args.model)
    print(f"Loading {args.pdf} …")
    reader.index(args.pdf)

    if args.page is not None:
        print(f"Summarizing page {args.page} …\n")
        summary = reader.summarize_page(args.page)
    else:
        print("Summarizing entire document …\n")
        summary = reader.summarize()

    print(f"Summary:\n{summary}")


def cmd_search(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    reader = PDFReader(api_key=args.api_key, model=args.model)

    if args.load:
        print(f"Loading index from {args.load} …")
        reader.load_index(args.load)
    else:
        print(f"Indexing {args.pdf} …")
        reader.index(args.pdf)

    print(f"\nSearching for: '{args.query}'\n")
    results = reader.search(args.query, k=args.top_k)
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        page = meta.get("page", "?")
        source = meta.get("source", "")
        score = result["score"]
        print(f"[{i}] Page {page} — {source}  (score: {score:.4f})")
        print(f"    {result['content'][:300].strip()}\n")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "index": cmd_index,
        "ask": cmd_ask,
        "summarize": cmd_summarize,
        "search": cmd_search,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
