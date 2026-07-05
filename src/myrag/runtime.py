from __future__ import annotations

import os
from pathlib import Path

import yaml

from myrag.config import AppSettings
from myrag.control import ConversationOrchestrator
from myrag.corpus import corpus_stats, load_markdown_corpus
from myrag.models import BenchmarkSuite, RetrievalPipelineSpec
from myrag.rag import ExperimentRunner, RAGEngine, default_experiment_registry


class Runtime:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.corpus_documents = load_markdown_corpus(settings.sample_corpus_dir)
        self.stats = corpus_stats(self.corpus_documents)
        self.spec = RetrievalPipelineSpec(name="baseline-dense", retriever_params={"top_k": settings.default_top_k})
        runtime_artifacts = settings.artifacts_dir / "runtime" / f"pid-{os.getpid()}"
        self.rag_engine = RAGEngine(self.spec, runtime_artifacts)
        self.rag_engine.build(self.corpus_documents)
        self.orchestrator = ConversationOrchestrator(self.rag_engine)
        self.benchmark_suite = load_benchmark_suite(settings.benchmark_path)
        self.experiment_runner = ExperimentRunner(self.corpus_documents, settings.artifacts_dir)


def load_benchmark_suite(path: Path) -> BenchmarkSuite:
    return BenchmarkSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def format_experiment_table(results: list) -> str:
    lines = ["run_id | hit_rate | mrr | ndcg | top1 | latency_ms", "--- | --- | --- | --- | --- | ---"]
    for item in results:
        lines.append(f"{item.run_id} | {item.metrics['hit_rate']:.3f} | {item.metrics['mrr']:.3f} | {item.metrics['ndcg']:.3f} | {item.metrics['top1_hit_rate']:.3f} | {item.latency['latency_ms']:.2f}")
    return "\n".join(lines)


def registry_summary() -> dict[str, dict]:
    return {name: spec.model_dump(mode="python") for name, spec in default_experiment_registry().items()}
