#!/usr/bin/env python3
"""LLMPDF – Command-line interface for the multi-source RAG knowledge base.

Examples
--------
Index a PDF and save the index:
    python main.py index document.pdf --save ./my_index

Ingest multiple sources at once:
    python main.py ingest --pdf paper.pdf --md ./notes --sql sqlite:///data.db --sql-table articles --save ./kb

Ask a question (with cross-encoder reranking):
    python main.py ask "What did the paper conclude?" --load ./kb --rerank

Run the retrieval eval:
    python main.py eval examples/eval_dataset.json --load ./kb
"""

import argparse
import sys
from typing import Optional

from src.config import Config
from src.knowledge_base import KnowledgeBase
from src.pdf_reader import PDFReader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmpdf",
        description=(
            "Multi-source RAG knowledge base: PDFs, markdown notes, and SQL, "
            "with optional cross-encoder reranking."
        ),
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

    # ---------------------------------------------------------------- index
    p_index = sub.add_parser(
        "index", help="Index a single PDF file (legacy convenience command)."
    )
    p_index.add_argument("pdf", help="Path to the PDF file.")
    p_index.add_argument("--save", metavar="DIR", help="Where to save the index.")

    # --------------------------------------------------------------- ingest
    p_ing = sub.add_parser(
        "ingest", help="Ingest one or more sources into a single knowledge base."
    )
    p_ing.add_argument(
        "--pdf", action="append", default=[], help="PDF file (repeatable)."
    )
    p_ing.add_argument(
        "--md",
        action="append",
        default=[],
        help="Markdown file or directory (repeatable).",
    )
    p_ing.add_argument(
        "--sql",
        help="SQLAlchemy connection URL, e.g. sqlite:///data.db",
    )
    p_ing.add_argument("--sql-table", help="Table name to ingest.")
    p_ing.add_argument("--sql-query", help="Custom SELECT query to ingest.")
    p_ing.add_argument(
        "--sql-id-column", help="Column to record as row_id in metadata."
    )
    p_ing.add_argument(
        "--sql-row-limit", type=int, help="Maximum rows to ingest."
    )
    p_ing.add_argument("--save", metavar="DIR", help="Where to save the index.")

    # ------------------------------------------------------------------ ask
    p_ask = sub.add_parser("ask", help="Ask a question against an indexed KB.")
    p_ask.add_argument("question", help="Question to ask.")
    p_ask.add_argument("--pdf", help="One-shot: ingest this PDF before asking.")
    p_ask.add_argument(
        "--load", metavar="DIR", help="Load a saved index from this directory."
    )
    p_ask.add_argument(
        "--sources", action="store_true", help="Print the source chunks used."
    )
    p_ask.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking to retrieved candidates.",
    )
    p_ask.add_argument(
        "--top-k",
        type=int,
        default=Config.SEARCH_TOP_K,
        help="Number of context chunks (default: %(default)s).",
    )

    # ------------------------------------------------------------- summarize
    p_sum = sub.add_parser("summarize", help="Summarize a PDF document.")
    p_sum.add_argument("pdf", help="Path to the PDF file.")
    p_sum.add_argument(
        "--page", type=int, default=None, metavar="N", help="Page (0-based) to summarize."
    )

    # ----------------------------------------------------------------- search
    p_search = sub.add_parser("search", help="Semantic search over the KB.")
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("pdf", nargs="?", help="PDF to index on the fly.")
    p_search.add_argument(
        "--load", metavar="DIR", help="Load a saved index from this directory."
    )
    p_search.add_argument(
        "--rerank", action="store_true", help="Apply cross-encoder reranking."
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=Config.SEARCH_TOP_K,
        help="Number of results (default: %(default)s).",
    )

    # ------------------------------------------------------------------- eval
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate retrieval quality (with and without reranking).",
    )
    p_eval.add_argument("dataset", help="Path to a JSON eval dataset.")
    p_eval.add_argument(
        "--load", metavar="DIR", help="Load a saved index from this directory."
    )
    p_eval.add_argument(
        "--top-k",
        type=int,
        default=Config.SEARCH_TOP_K,
        help="Top-k for hit@k (default: %(default)s).",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_api_key(api_key: Optional[str]) -> None:
    if not api_key:
        print(
            "Error: OpenAI API key is required. "
            "Set OPENAI_API_KEY in your .env file or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)


def _print_sources(sources) -> None:
    print("Sources:")
    for i, src in enumerate(sources, 1):
        meta = src["metadata"]
        label = meta.get("source", "?")
        page = meta.get("page")
        extra = f" (page {page})" if page is not None else ""
        print(f"  [{i}] {label}{extra}")
        snippet = src["content"][:200].strip().replace("\n", " ")
        print(f"      {snippet} …")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    reader = PDFReader(api_key=args.api_key, model=args.model)
    print(f"Indexing {args.pdf} …")
    n = reader.index(args.pdf)
    print(f"✓ Indexed {n} chunks from {args.pdf}.")
    if args.save:
        reader.save_index(args.save)
        print(f"✓ Index saved to {args.save}.")


def cmd_ingest(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    if not (args.pdf or args.md or args.sql):
        print("Error: provide at least one of --pdf, --md, --sql.", file=sys.stderr)
        sys.exit(1)

    kb = KnowledgeBase(api_key=args.api_key, model=args.model)
    total = 0

    for pdf_path in args.pdf:
        print(f"Ingesting PDF: {pdf_path}")
        total += kb.add_pdf(pdf_path)

    for md_path in args.md:
        print(f"Ingesting Markdown: {md_path}")
        total += kb.add_markdown(md_path)

    if args.sql:
        if not (args.sql_table or args.sql_query):
            print(
                "Error: --sql requires --sql-table or --sql-query.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Ingesting SQL: {args.sql}")
        total += kb.add_sql(
            connection_url=args.sql,
            table=args.sql_table,
            query=args.sql_query,
            id_column=args.sql_id_column,
            row_limit=args.sql_row_limit,
        )

    print(f"✓ Total chunks indexed: {total}")
    if args.save:
        kb.save_index(args.save)
        print(f"✓ Index saved to {args.save}.")


def cmd_ask(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    kb = KnowledgeBase(
        api_key=args.api_key, model=args.model, use_reranker=args.rerank
    )

    if args.load:
        print(f"Loading index from {args.load} …")
        kb.load_index(args.load)
    elif args.pdf:
        print(f"Indexing {args.pdf} …")
        kb.add_pdf(args.pdf)
    else:
        print(
            "Error: provide either --pdf or --load.", file=sys.stderr
        )
        sys.exit(1)

    if args.sources:
        result = kb.ask_with_sources(args.question, k=args.top_k)
        print(f"\nAnswer:\n{result['answer']}\n")
        _print_sources(result["sources"])
    else:
        answer = kb.ask(args.question, k=args.top_k)
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
    kb = KnowledgeBase(
        api_key=args.api_key, model=args.model, use_reranker=args.rerank
    )

    if args.load:
        print(f"Loading index from {args.load} …")
        kb.load_index(args.load)
    elif args.pdf:
        print(f"Indexing {args.pdf} …")
        kb.add_pdf(args.pdf)
    else:
        print("Error: provide either a PDF path or --load.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSearching for: '{args.query}'\n")
    results = kb.search(args.query, k=args.top_k)
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        label = meta.get("source", "?")
        page = meta.get("page")
        score = result["score"]
        extra = f" page {page}" if page is not None else ""
        print(f"[{i}] {label}{extra}  (score: {score:.4f})")
        print(f"    {result['content'][:300].strip()}\n")


def cmd_eval(args: argparse.Namespace) -> None:
    _require_api_key(args.api_key)
    from src.eval import run_eval

    run_eval(
        dataset_path=args.dataset,
        index_dir=args.load,
        api_key=args.api_key,
        top_k=args.top_k,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "index": cmd_index,
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "summarize": cmd_summarize,
        "search": cmd_search,
        "eval": cmd_eval,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
