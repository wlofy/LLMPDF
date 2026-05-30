"""Retrieval evaluation: hit@k with and without cross-encoder reranking.

Dataset format (JSON list)::

    [
      {
        "question": "What is FAISS?",
        "keywords": ["FAISS", "vector"],     # ANY of these in the chunk = hit
        "source_contains": "vector_store"     # optional substring of source path
      },
      ...
    ]

A chunk is counted as relevant if all provided constraints match. Hit@k is the
fraction of questions where at least one of the top-k retrieved chunks is
relevant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .knowledge_base import KnowledgeBase


def _is_relevant(chunk: Dict[str, Any], q: Dict[str, Any]) -> bool:
    content = chunk["content"].lower()
    meta_source = str(chunk["metadata"].get("source", "")).lower()

    keywords = q.get("keywords") or []
    if keywords and not any(kw.lower() in content for kw in keywords):
        return False

    needle = q.get("source_contains")
    if needle and needle.lower() not in meta_source:
        return False

    return True


def _hit_at_k(results: List[Dict[str, Any]], q: Dict[str, Any]) -> bool:
    return any(_is_relevant(r, q) for r in results)


def _hits_at_k_curve(
    kb: KnowledgeBase, dataset: List[Dict[str, Any]], top_k: int
) -> Dict[str, float]:
    """Return hit@1, hit@3, hit@k averaged over the dataset."""
    ks = sorted({1, min(3, top_k), top_k})
    counts = {k: 0 for k in ks}

    for q in dataset:
        results = kb.search(q["question"], k=top_k)
        for k in ks:
            if _hit_at_k(results[:k], q):
                counts[k] += 1

    n = max(1, len(dataset))
    return {f"hit@{k}": counts[k] / n for k in ks}


def run_eval(
    dataset_path: str,
    index_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Run the eval twice — without and with reranking — and print results.

    Returns:
        Dict with ``baseline`` and ``reranked`` keys, each mapping ``hit@k``
        labels to scores in ``[0, 1]``.
    """
    path = Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {dataset_path}")
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Eval dataset must be a non-empty JSON list.")

    print(f"Running eval over {len(dataset)} questions (top-k={top_k})\n")

    print("→ Baseline (FAISS only)")
    kb_base = KnowledgeBase(api_key=api_key, use_reranker=False)
    if index_dir:
        kb_base.load_index(index_dir)
    else:
        raise ValueError("--load is required: point eval at a saved index.")
    baseline = _hits_at_k_curve(kb_base, dataset, top_k)
    for k_label, score in baseline.items():
        print(f"   {k_label}: {score:.3f}")

    print("\n→ With cross-encoder reranker")
    kb_rer = KnowledgeBase(api_key=api_key, use_reranker=True)
    kb_rer.load_index(index_dir)
    reranked = _hits_at_k_curve(kb_rer, dataset, top_k)
    for k_label, score in reranked.items():
        print(f"   {k_label}: {score:.3f}")

    print("\n→ Delta (reranked − baseline)")
    for k_label in baseline:
        delta = reranked[k_label] - baseline[k_label]
        sign = "+" if delta >= 0 else ""
        print(f"   {k_label}: {sign}{delta:.3f}")

    return {"baseline": baseline, "reranked": reranked}
