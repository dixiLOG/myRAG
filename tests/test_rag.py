from pathlib import Path

from myrag.corpus import load_markdown_corpus
from myrag.models import RetrievalPipelineSpec
from myrag.rag import RAGEngine
from myrag.runtime import load_benchmark_suite


def test_rag_engine_retrieves_expected_document(tmp_path: Path) -> None:
    corpus_root = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"
    benchmark_path = Path(__file__).resolve().parents[1] / "benchmarks" / "sample_benchmark.yaml"
    documents = load_markdown_corpus(corpus_root)
    engine = RAGEngine(RetrievalPipelineSpec(name="test-dense"), tmp_path)
    engine.build(documents)
    answer = engine.answer("飞书文档下载为 Markdown 的配置命令是什么？")
    assert answer.citations
    assert answer.citations[0].source_id == "feishu_github_tutorial"
    result = engine.run_benchmark(load_benchmark_suite(benchmark_path))
    assert result.metrics["hit_rate"] >= 0.7
    assert result.metrics["citation_completeness"] >= 0.7
