from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

IntentName = Literal["knowledge_qa", "clarify", "simple_chat"]
ActionType = Literal["follow_up", "view_source", "narrow_query"]
Visibility = Literal["public", "private", "draft"]
DocType = Literal["essay", "tutorial", "longform", "book_note", "note", "unknown"]
UIBlockType = Literal["text", "citations", "next_actions", "notice"]


class CorpusDocumentMeta(BaseModel):
    doc_id: str
    title: str
    source_path: str
    doc_type: DocType = "unknown"
    tags: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    language: str = "zh"
    visibility: Visibility = "public"
    series: str | None = None
    authors: list[str] = Field(default_factory=list)

    def as_metadata_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


class CorpusDocument(BaseModel):
    meta: CorpusDocumentMeta
    text: str


class RetrievalResult(BaseModel):
    node_id: str
    source_id: str
    doc_title: str
    source_path: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0
    dense_score: float = 0.0
    lexical_score: float = 0.0
    visibility: Visibility = "public"


class CitationItem(BaseModel):
    source_id: str
    doc_title: str
    source_path: str
    chunk_id: str
    excerpt: str
    relevance_score: float
    visibility: Visibility = "public"


class NextActionItem(BaseModel):
    action_type: ActionType
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)
    reason: str


class UIBlock(BaseModel):
    block_type: UIBlockType
    title: str | None = None
    content: Any


class IntentClassification(BaseModel):
    intent: IntentName
    confidence: float
    route_reason: str


class CheckpointRecord(BaseModel):
    checkpoint_id: str
    label: str
    snapshot: dict[str, Any]
    allows_rollback: bool = True


class RollbackDecision(BaseModel):
    allowed: bool
    target_checkpoint: str | None = None
    reason: str


class RetrievalPipelineSpec(BaseModel):
    name: str = "baseline-dense"
    embedding_model: str = "deterministic-hash-v1"
    chunk_strategy: str = "markdown-structure-v1"
    metadata_policy: str = "default-v1"
    retriever_type: str = "dense"
    retriever_params: dict[str, Any] = Field(default_factory=lambda: {"top_k": 5})
    reranker_type: str = "none"
    reranker_params: dict[str, Any] = Field(default_factory=dict)
    response_mode: str = "compact_citations"
    chunk_size: int = 420
    chunk_overlap: int = 60

    @property
    def index_version(self) -> str:
        payload = self.model_dump_json(indent=None)
        return sha1(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def embedding_version(self) -> str:
        return f"{self.embedding_model}:{self.chunk_strategy}"

    @property
    def retrieval_version(self) -> str:
        return f"{self.retriever_type}:{self.reranker_type}"


class BenchmarkCase(BaseModel):
    case_id: str
    question: str
    content_type: str
    query_type: str
    expected_doc_ids: list[str]
    expected_phrases: list[str] = Field(default_factory=list)


class BenchmarkSuite(BaseModel):
    name: str
    cases: list[BenchmarkCase]


class ExperimentResult(BaseModel):
    run_id: str
    corpus_version: str
    index_version: str
    embedding_version: str
    retrieval_version: str
    rerank_version: str
    metrics: dict[str, float]
    cost: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class AnswerBundle(BaseModel):
    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConversationState(BaseModel):
    thread_id: str
    messages: list[ConversationMessage] = Field(default_factory=list)
    intent: IntentName = "clarify"
    intent_confidence: float = 0.0
    user_goal: str = ""
    active_subgraph: str = "clarify_subgraph"
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    draft_answer: str = ""
    citations: list[CitationItem] = Field(default_factory=list)
    next_actions: list[NextActionItem] = Field(default_factory=list)
    ui_blocks: list[UIBlock] = Field(default_factory=list)
    step_count: int = 0
    current_checkpoint_ref: str | None = None
    rollback_target: str | None = None
    last_error: str | None = None
    safety_flags: list[str] = Field(default_factory=list)
    checkpoints: list[CheckpointRecord] = Field(default_factory=list)
    final_response: str = ""

    def latest_user_message(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content.strip()
        return ""


class QueryRequest(BaseModel):
    question: str
    history: list[ConversationMessage] = Field(default_factory=list)


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    intent: IntentName
    confidence: float
    citations: list[CitationItem]
    next_actions: list[NextActionItem]
    ui_blocks: list[UIBlock]
    rollback_target: str | None = None
    safety_flags: list[str] = Field(default_factory=list)


def corpus_version_for_documents(documents: list[CorpusDocument]) -> str:
    payload = "\n".join(
        f"{doc.meta.doc_id}|{doc.meta.updated_at}|{len(doc.text)}"
        for doc in sorted(documents, key=lambda item: item.meta.doc_id)
    )
    return sha1(payload.encode("utf-8")).hexdigest()[:12]


def safe_slug(value: str) -> str:
    return sha1(value.encode("utf-8")).hexdigest()[:12]


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()

