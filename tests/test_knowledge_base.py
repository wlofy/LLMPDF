"""Tests for the multi-source KnowledgeBase facade."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.knowledge_base import KnowledgeBase


def _patched_kb():
    """Return a KB with VectorStore + LLMInterface stubbed out."""
    with patch("src.knowledge_base.VectorStore") as MockVS, \
         patch("src.knowledge_base.LLMInterface") as MockLLM:
        vs_instance = MockVS.return_value
        vs_instance.is_ready = False
        llm_instance = MockLLM.return_value
        kb = KnowledgeBase(api_key="sk-test")
    return kb, vs_instance, llm_instance


class TestIngestion:
    def test_add_markdown_builds_then_adds(self, tmp_path):
        (tmp_path / "a.md").write_text("alpha content " * 30)
        (tmp_path / "b.md").write_text("beta content " * 30)

        with patch("src.knowledge_base.VectorStore") as MockVS, \
             patch("src.knowledge_base.LLMInterface"):
            vs = MockVS.return_value
            # Simulate is_ready flipping after first build.
            vs.is_ready = False

            def _build(_chunks):
                vs.is_ready = True

            vs.build.side_effect = _build

            kb = KnowledgeBase(api_key="sk-test", chunk_size=100, chunk_overlap=10)
            n1 = kb.add_markdown(str(tmp_path))
            n2 = kb.add_markdown(str(tmp_path))

        assert n1 > 0
        assert n2 > 0
        vs.build.assert_called_once()
        vs.add_documents.assert_called_once()

    def test_add_pdf_uses_pdf_loader(self, tmp_path):
        pdf_file = tmp_path / "x.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        with patch("src.knowledge_base.VectorStore") as MockVS, \
             patch("src.knowledge_base.LLMInterface"), \
             patch("src.loaders.PyPDFLoader") as MockPDF:
            MockVS.return_value.is_ready = False
            MockPDF.return_value.load.return_value = [
                Document(page_content="hello world " * 30, metadata={"page": 0})
            ]
            kb = KnowledgeBase(api_key="sk-test", chunk_size=100, chunk_overlap=10)
            n = kb.add_pdf(str(pdf_file))

        assert n >= 1
        MockPDF.assert_called_once()


class TestReranker:
    def test_use_reranker_flag_creates_reranker(self):
        with patch("src.knowledge_base.VectorStore"), \
             patch("src.knowledge_base.LLMInterface") as MockLLM, \
             patch("src.knowledge_base.CrossEncoderReranker") as MockRer:
            KnowledgeBase(api_key="sk-test", use_reranker=True)

        MockRer.assert_called_once()
        kwargs = MockLLM.call_args.kwargs
        assert kwargs["reranker"] is MockRer.return_value

    def test_no_reranker_by_default(self):
        with patch("src.knowledge_base.VectorStore"), \
             patch("src.knowledge_base.LLMInterface") as MockLLM, \
             patch("src.knowledge_base.CrossEncoderReranker") as MockRer:
            KnowledgeBase(api_key="sk-test")

        MockRer.assert_not_called()
        assert MockLLM.call_args.kwargs["reranker"] is None


class TestQuestionAnswering:
    def test_ask_returns_answer_string(self):
        kb, _, llm = _patched_kb()
        llm.answer.return_value = {"result": "the answer", "source_documents": []}
        assert kb.ask("q?") == "the answer"

    def test_ask_with_sources_includes_metadata(self):
        kb, _, llm = _patched_kb()
        doc = Document(page_content="ctx", metadata={"source": "/tmp/x.md"})
        llm.answer.return_value = {"result": "A", "source_documents": [doc]}

        out = kb.ask_with_sources("q?")
        assert out["answer"] == "A"
        assert out["sources"][0]["metadata"]["source"] == "/tmp/x.md"
