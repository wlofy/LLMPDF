"""Tests for CrossEncoderReranker (model layer is mocked to keep tests fast)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.reranker import CrossEncoderReranker


def _docs(texts):
    return [Document(page_content=t, metadata={"i": i}) for i, t in enumerate(texts)]


class TestReranker:
    def test_empty_input_returns_empty(self):
        r = CrossEncoderReranker()
        assert r.rerank("q", []) == []
        assert r.score("q", []) == []

    def test_rerank_orders_by_score_descending(self):
        r = CrossEncoderReranker()
        fake = MagicMock()
        # Scores chosen so the original ordering is reversed.
        fake.predict.return_value = [0.1, 0.9, 0.5]
        r._model = fake  # bypass model download

        docs = _docs(["lowest", "highest", "middle"])
        out = r.rerank("q", docs)

        assert [d.page_content for d in out] == ["highest", "middle", "lowest"]

    def test_rerank_top_k_truncates(self):
        r = CrossEncoderReranker()
        fake = MagicMock()
        fake.predict.return_value = [0.1, 0.9, 0.5, 0.7]
        r._model = fake

        docs = _docs(["a", "b", "c", "d"])
        out = r.rerank("q", docs, top_k=2)

        assert len(out) == 2
        assert [d.page_content for d in out] == ["b", "d"]

    def test_rerank_with_scores_returns_pairs(self):
        r = CrossEncoderReranker()
        fake = MagicMock()
        fake.predict.return_value = [0.2, 0.8]
        r._model = fake

        docs = _docs(["x", "y"])
        out = r.rerank_with_scores("q", docs)

        assert out[0][0].page_content == "y"
        assert out[0][1] == 0.8
        assert out[1][1] == 0.2

    def test_model_is_loaded_lazily(self):
        # Inject a fake sentence_transformers module so the lazy import
        # inside _ensure_model() resolves without the real dependency.
        fake_module = MagicMock()
        instance = MagicMock()
        instance.predict.return_value = [0.5]
        fake_module.CrossEncoder.return_value = instance

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            r = CrossEncoderReranker()
            assert r._model is None
            r.score("q", _docs(["only"]))

        fake_module.CrossEncoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
