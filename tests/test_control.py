from pathlib import Path

from myrag.config import AppSettings
from myrag.control import ConversationOrchestrator
from myrag.corpus import load_markdown_corpus
from myrag.models import RetrievalPipelineSpec
from myrag.rag import RAGEngine


def build_orchestrator(tmp_path: Path) -> ConversationOrchestrator:
    corpus_root = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"
    documents = load_markdown_corpus(corpus_root)
    engine = RAGEngine(RetrievalPipelineSpec(name="control-test"), tmp_path)
    engine.build(documents)
    return ConversationOrchestrator(engine)


def test_control_plane_returns_structured_citations_and_actions(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(tmp_path)
    response = orchestrator.run("外儒内法这篇文章最后用哪八个字总结方法论？")
    assert response.intent == "knowledge_qa"
    assert response.citations
    assert response.next_actions
    assert any(block.block_type == "citations" for block in response.ui_blocks)


def test_control_plane_can_fall_back_to_clarification(tmp_path: Path) -> None:
    orchestrator = build_orchestrator(tmp_path)
    response = orchestrator.run("hi")
    assert response.intent == "simple_chat"
    response2 = orchestrator.run("啊")
    assert response2.intent == "clarify"
