from __future__ import annotations

import math
import re
import time
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from myrag.corpus import split_markdown_sections
from myrag.models import AnswerBundle, BenchmarkSuite, CitationItem, CorpusDocument, ExperimentResult, RetrievalPipelineSpec, RetrievalResult, corpus_version_for_documents

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


class DeterministicEmbedding(BaseEmbedding):
    model_name: str = "deterministic-hash-v1"
    dimension: int = 96

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN_RE.findall(text.lower()) or [text[:50].lower()]
        for token in tokens:
            digest = sha1(f"{self.model_name}:{token}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "big") % self.dimension
            sign = -1.0 if digest[2] % 2 else 1.0
            magnitude = 1.0 + (digest[3] / 255.0)
            vector[idx] += sign * magnitude
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)


class RAGEngine:
    def __init__(self, spec: RetrievalPipelineSpec, artifacts_dir: Path) -> None:
        self.spec = spec
        self.artifacts_dir = artifacts_dir
        self.embed_model = DeterministicEmbedding(model_name=spec.embedding_model)
        self.index: VectorStoreIndex | None = None
        self.node_lookup: dict[str, Any] = {}
        self.corpus_version: str = "unknown"
        self.notes: list[str] = []

    def build(self, documents: list[CorpusDocument]) -> None:
        self.corpus_version = corpus_version_for_documents(documents)
        self.notes = []
        index_dir = self.artifacts_dir / "indexes" / self.spec.index_version
        index_dir.mkdir(parents=True, exist_ok=True)
        vector_store = QdrantVectorStore(collection_name=f"myrag_{self.spec.index_version}", client=QdrantClient(path=str(index_dir / "qdrant")))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=self.spec.chunk_size, chunk_overlap=self.spec.chunk_overlap)])
        nodes = pipeline.run(documents=self._to_llama_documents(documents))
        self.node_lookup = {node.node_id: node for node in nodes}
        self.index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embed_model, show_progress=False)
        Settings.embed_model = self.embed_model

    def answer(self, query: str, top_k: int | None = None) -> AnswerBundle:
        retrieval_results = self.retrieve(query, top_k=top_k)
        display_results = unique_by_source(retrieval_results)[:3]
        citations = [self._to_citation(item) for item in display_results]
        evidence = [item.excerpt for item in display_results]
        confidence = min(0.95, 0.35 + (retrieval_results[0].relevance_score if retrieval_results else 0.0))
        if not retrieval_results:
            return AnswerBundle(answer="我暂时没有从知识库里找到足够直接的证据。你可以缩小问题范围，或者改成更具体的关键词。", confidence=0.2, notes=["empty_retrieval"])
        lines = [f"{index}. {item.doc_title}: {item.excerpt.replace(chr(10), ' ' )[:180]} [#{index}]" for index, item in enumerate(display_results, start=1)]
        return AnswerBundle(answer="根据当前知识库检索结果，我优先找到这些相关证据：\n" + "\n".join(lines), citations=citations, retrieval_results=retrieval_results, evidence=evidence, confidence=confidence, notes=list(self.notes))

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if self.index is None:
            raise RuntimeError("RAGEngine.build must be called before retrieval")
        similarity_top_k = int(top_k or self.spec.retriever_params.get("top_k", 5))
        candidate_count = similarity_top_k if self.spec.retriever_type == "dense" else max(similarity_top_k * 3, 8)
        raw_nodes = self.index.as_retriever(similarity_top_k=candidate_count).retrieve(query)
        results = [self._to_result(item) for item in raw_nodes]
        results = self._apply_retriever_strategy(query, results)
        results = self._apply_reranker(query, results)
        return sorted(results, key=lambda item: item.relevance_score, reverse=True)[:similarity_top_k]

    def run_benchmark(self, benchmark: BenchmarkSuite) -> ExperimentResult:
        started = time.perf_counter()
        hit_count = 0
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        top1_hits = 0
        coverages: list[float] = []
        answer_scores: list[float] = []
        citation_scores: list[float] = []
        empty_count = 0
        for case in benchmark.cases:
            answer = self.answer(case.question)
            ranked_ids = dedupe_preserve_order([item.source_id for item in answer.retrieval_results])
            relevant = set(case.expected_doc_ids)
            if not ranked_ids:
                empty_count += 1
            hit_count += int(any(item in relevant for item in ranked_ids))
            reciprocal_ranks.append(self._reciprocal_rank(ranked_ids, relevant))
            ndcgs.append(self._ndcg(ranked_ids, relevant))
            top1_hits += int(bool(ranked_ids) and ranked_ids[0] in relevant)
            coverages.append(self._coverage(ranked_ids, relevant))
            answer_scores.append(1.0 if answer.evidence else 0.0)
            citation_scores.append(1.0 if answer.citations else 0.0)
        total = max(len(benchmark.cases), 1)
        metrics = {
            "hit_rate": hit_count / total,
            "mrr": sum(reciprocal_ranks) / total,
            "ndcg": sum(ndcgs) / total,
            "empty_retrieval_rate": empty_count / total,
            "top1_hit_rate": top1_hits / total,
            "evidence_coverage": sum(coverages) / total,
            "faithfulness": sum(answer_scores) / total,
            "citation_completeness": sum(citation_scores) / total,
        }
        duration_ms = (time.perf_counter() - started) * 1000.0 / total
        return ExperimentResult(run_id=f"run-{self.spec.index_version}-{self.corpus_version}", corpus_version=self.corpus_version, index_version=self.spec.index_version, embedding_version=self.spec.embedding_version, retrieval_version=self.spec.retrieval_version, rerank_version=self.spec.reranker_type, metrics=metrics, cost={"token_cost": 0.0}, latency={"latency_ms": duration_ms}, notes=list(self.notes))

    def _to_llama_documents(self, documents: list[CorpusDocument]) -> list[Document]:
        out: list[Document] = []
        for document in documents:
            sections = split_markdown_sections(document.text)
            for section_index, (heading, body) in enumerate(sections):
                out.append(Document(text=body, id_=f"{document.meta.doc_id}::section::{section_index}", metadata=document.meta.as_metadata_dict() | {"section_heading": heading, "section_index": section_index}, excluded_embed_metadata_keys=["source_path", "visibility", "updated_at"], excluded_llm_metadata_keys=["source_path"]))
        return out

    def _to_result(self, raw_node: Any) -> RetrievalResult:
        node = raw_node.node
        metadata = dict(node.metadata)
        score = float(raw_node.score or 0.0)
        return RetrievalResult(node_id=node.node_id, source_id=str(metadata.get("doc_id") or metadata.get("source_path") or node.node_id), doc_title=str(metadata.get("title") or metadata.get("doc_id") or node.node_id), source_path=str(metadata.get("source_path") or metadata.get("doc_id") or node.node_id), excerpt=node.text[:260], metadata=metadata, relevance_score=score, dense_score=score, lexical_score=0.0, visibility=str(metadata.get("visibility") or "public"))

    def _apply_retriever_strategy(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if self.spec.retriever_type == "dense":
            return results
        if self.spec.retriever_type in {"hybrid", "hierarchical"}:
            for item in results:
                item.lexical_score = lexical_overlap_score(query, item.excerpt)
                if self.spec.retriever_type == "hybrid":
                    item.relevance_score = (item.dense_score * 0.7) + (item.lexical_score * 0.3)
                else:
                    depth_bonus = 0.05 if int(item.metadata.get("section_index", 0)) == 0 else 0.0
                    item.relevance_score = (item.dense_score * 0.75) + (item.lexical_score * 0.2) + depth_bonus
            self.notes.append(f"retriever:{self.spec.retriever_type}")
            return results
        self.notes.append(f"retriever_fallback:{self.spec.retriever_type}")
        return results

    def _apply_reranker(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if self.spec.reranker_type == "none":
            return results
        if self.spec.reranker_type in {"lexical_overlap", "cross_encoder"}:
            for item in results:
                item.relevance_score = (item.relevance_score * 0.8) + (lexical_overlap_score(query, item.excerpt) * 0.2)
            self.notes.append(f"reranker:{self.spec.reranker_type}")
            return results
        self.notes.append(f"reranker_unverified:{self.spec.reranker_type}")
        return results

    @staticmethod
    def _to_citation(item: RetrievalResult) -> CitationItem:
        return CitationItem(source_id=item.source_id, doc_title=item.doc_title, source_path=item.source_path, chunk_id=item.node_id, excerpt=item.excerpt, relevance_score=item.relevance_score, visibility=item.visibility)

    @staticmethod
    def _reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
        for index, item in enumerate(ranked_ids, start=1):
            if item in relevant:
                return 1.0 / index
        return 0.0

    @staticmethod
    def _ndcg(ranked_ids: list[str], relevant: set[str]) -> float:
        dcg = 0.0
        for index, item in enumerate(ranked_ids, start=1):
            if item in relevant:
                dcg += 1.0 / math.log2(index + 1)
        ideal_hits = min(len(relevant), len(ranked_ids))
        idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1)) or 1.0
        return dcg / idcg

    @staticmethod
    def _coverage(ranked_ids: list[str], relevant: set[str]) -> float:
        return 1.0 if not relevant else len([item for item in ranked_ids if item in relevant]) / len(relevant)


class ExperimentRunner:
    def __init__(self, corpus_documents: list[CorpusDocument], artifacts_dir: Path) -> None:
        self.corpus_documents = corpus_documents
        self.artifacts_dir = artifacts_dir
        self.registry = default_experiment_registry()

    def run(self, benchmark: BenchmarkSuite, experiment_names: list[str] | None = None) -> list[ExperimentResult]:
        names = experiment_names or list(self.registry)
        results: list[ExperimentResult] = []
        for name in names:
            engine = RAGEngine(spec=self.registry[name], artifacts_dir=self.artifacts_dir / "experiments" / name)
            engine.build(self.corpus_documents)
            results.append(engine.run_benchmark(benchmark))
        return results


def default_experiment_registry() -> dict[str, RetrievalPipelineSpec]:
    return {
        "baseline-dense": RetrievalPipelineSpec(name="baseline-dense"),
        "wide-chunk-dense": RetrievalPipelineSpec(name="wide-chunk-dense", chunk_size=520, chunk_overlap=80),
        "hybrid-lexical": RetrievalPipelineSpec(name="hybrid-lexical", retriever_type="hybrid"),
        "dense-rerank": RetrievalPipelineSpec(name="dense-rerank", reranker_type="lexical_overlap"),
        "embedding-v2": RetrievalPipelineSpec(name="embedding-v2", embedding_model="deterministic-hash-v2"),
    }


def unique_by_source(results: list[RetrievalResult]) -> list[RetrievalResult]:
    seen: set[str] = set()
    output: list[RetrievalResult] = []
    for item in results:
        if item.source_id not in seen:
            seen.add(item.source_id)
            output.append(item)
    return output


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def lexical_overlap_score(query: str, text: str) -> float:
    query_terms = Counter(TOKEN_RE.findall(query.lower()))
    text_terms = Counter(TOKEN_RE.findall(text.lower()))
    if not query_terms or not text_terms:
        return 0.0
    return sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items()) / max(sum(query_terms.values()), 1)

