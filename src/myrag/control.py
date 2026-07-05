from __future__ import annotations

from copy import deepcopy
from typing import Protocol
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from myrag.models import AnswerBundle, CheckpointRecord, CitationItem, ConversationMessage, ConversationState, IntentClassification, NextActionItem, QueryResponse, RollbackDecision, UIBlock
from myrag.rag import RAGEngine


class GraphState(TypedDict, total=False):
    thread_id: str
    messages: list[ConversationMessage]
    intent: str
    intent_confidence: float
    user_goal: str
    active_subgraph: str
    retrieval_results: list
    evidence: list[str]
    draft_answer: str
    citations: list[CitationItem]
    next_actions: list[NextActionItem]
    ui_blocks: list[UIBlock]
    step_count: int
    current_checkpoint_ref: str | None
    rollback_target: str | None
    last_error: str | None
    safety_flags: list[str]
    checkpoints: list[CheckpointRecord]
    final_response: str


class IntentClassifier(Protocol):
    def classify(self, state: ConversationState) -> IntentClassification: ...


class CitationFormatter(Protocol):
    def format(self, citations: list[CitationItem]) -> list[UIBlock]: ...


class NextActionGenerator(Protocol):
    def generate(self, state: ConversationState) -> list[NextActionItem]: ...


class RollbackPolicy(Protocol):
    def decide(self, state: ConversationState) -> RollbackDecision: ...
    def apply(self, state: ConversationState, decision: RollbackDecision) -> ConversationState: ...


class RuleBasedIntentClassifier:
    def classify(self, state: ConversationState) -> IntentClassification:
        latest = state.latest_user_message().lower()
        if latest in {"hi", "hello", "你好", "嗨"}:
            return IntentClassification(intent="simple_chat", confidence=0.9, route_reason="greeting")
        if not latest or len(latest) < 3:
            return IntentClassification(intent="clarify", confidence=0.2, route_reason="question_too_short")
        if any(token in latest for token in ["怎么", "如何", "为什么", "哪些", "总结", "命令", "步骤", "什么", "换个思路", "引用"]):
            return IntentClassification(intent="knowledge_qa", confidence=0.88, route_reason="knowledge_pattern")
        if latest.endswith("?") or latest.endswith("？"):
            return IntentClassification(intent="knowledge_qa", confidence=0.75, route_reason="question_mark")
        return IntentClassification(intent="knowledge_qa", confidence=0.6, route_reason="default_knowledge_route")


class DefaultCitationFormatter:
    def format(self, citations: list[CitationItem]) -> list[UIBlock]:
        if not citations:
            return [UIBlock(block_type="notice", title="No Citations", content="当前回答没有足够引用。")]
        inline = [f"[{index}] {item.doc_title}" for index, item in enumerate(citations, start=1)]
        cards = [{"index": index, "title": item.doc_title, "source_path": item.source_path, "excerpt": item.excerpt, "score": round(item.relevance_score, 4)} for index, item in enumerate(citations, start=1)]
        return [UIBlock(block_type="text", title="Inline Citations", content=" ".join(inline)), UIBlock(block_type="citations", title="Source Cards", content=cards)]


class DefaultNextActionGenerator:
    def generate(self, state: ConversationState) -> list[NextActionItem]:
        actions: list[NextActionItem] = [NextActionItem(action_type="follow_up", label="继续追问", payload={"question": f"基于『{state.latest_user_message()[:24]}』继续追问"}, reason="当前结果已经有可继续展开的证据。")]
        if state.citations:
            first = state.citations[0]
            actions.append(NextActionItem(action_type="view_source", label="查看原文", payload={"source_path": first.source_path, "chunk_id": first.chunk_id}, reason="先回到最相关原文，确认上下文。"))
        actions.append(NextActionItem(action_type="narrow_query", label="缩小问题", payload={"hint": "增加文章标题、人物、命令或章节名"}, reason="问题越具体，召回和排序越稳。"))
        return actions[:4]


class DefaultRollbackPolicy:
    def decide(self, state: ConversationState) -> RollbackDecision:
        latest = state.latest_user_message()
        if "换个思路" in latest and state.checkpoints:
            return RollbackDecision(allowed=True, target_checkpoint=state.checkpoints[-1].checkpoint_id, reason="user_requested_alternate_route")
        if state.intent == "knowledge_qa" and state.intent_confidence < 0.5 and state.checkpoints:
            return RollbackDecision(allowed=True, target_checkpoint=state.checkpoints[-1].checkpoint_id, reason="low_intent_confidence")
        if state.intent == "knowledge_qa" and not state.citations and state.checkpoints:
            return RollbackDecision(allowed=True, target_checkpoint=state.checkpoints[-1].checkpoint_id, reason="missing_citations")
        return RollbackDecision(allowed=False, reason="no_rollback_needed")

    def apply(self, state: ConversationState, decision: RollbackDecision) -> ConversationState:
        if not decision.allowed or not decision.target_checkpoint:
            return state
        checkpoint = next((item for item in reversed(state.checkpoints) if item.checkpoint_id == decision.target_checkpoint), None)
        if checkpoint is None:
            state.last_error = f"rollback_target_missing:{decision.target_checkpoint}"
            return state
        restored = ConversationState.model_validate(checkpoint.snapshot)
        restored.checkpoints = state.checkpoints
        restored.rollback_target = decision.target_checkpoint
        restored.last_error = f"rollback:{decision.reason}"
        restored.safety_flags = list(dict.fromkeys([*restored.safety_flags, "rollback_applied"]))
        restored.draft_answer = "我先回退到上一个可解释状态。你可以补充更具体的线索，或者直接查看最相关原文。"
        return restored


class ConversationOrchestrator:
    def __init__(self, rag_engine: RAGEngine, classifier: IntentClassifier | None = None, citation_formatter: CitationFormatter | None = None, next_action_generator: NextActionGenerator | None = None, rollback_policy: RollbackPolicy | None = None) -> None:
        self.rag_engine = rag_engine
        self.classifier = classifier or RuleBasedIntentClassifier()
        self.citation_formatter = citation_formatter or DefaultCitationFormatter()
        self.next_action_generator = next_action_generator or DefaultNextActionGenerator()
        self.rollback_policy = rollback_policy or DefaultRollbackPolicy()
        self.graph = self._build_graph()

    def run(self, question: str, history: list[ConversationMessage] | None = None, thread_id: str | None = None) -> QueryResponse:
        initial_state = ConversationState(thread_id=thread_id or str(uuid4()), messages=[*(history or []), ConversationMessage(role="user", content=question)])
        final_state = ConversationState.model_validate(self.graph.invoke(initial_state.model_dump(mode="python"), config={"configurable": {"thread_id": initial_state.thread_id}}))
        return QueryResponse(thread_id=final_state.thread_id, answer=final_state.final_response, intent=final_state.intent, confidence=final_state.intent_confidence, citations=final_state.citations, next_actions=final_state.next_actions, ui_blocks=final_state.ui_blocks, rollback_target=final_state.rollback_target, safety_flags=final_state.safety_flags)

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("receive_turn", self._receive_turn)
        builder.add_node("intent_router", self._intent_router)
        builder.add_node("policy_gate", self._policy_gate)
        builder.add_node("execute_subgraph", self._execute_subgraph)
        builder.add_node("response_postprocess", self._response_postprocess)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "receive_turn")
        builder.add_edge("receive_turn", "intent_router")
        builder.add_edge("intent_router", "policy_gate")
        builder.add_edge("policy_gate", "execute_subgraph")
        builder.add_edge("execute_subgraph", "response_postprocess")
        builder.add_edge("response_postprocess", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=InMemorySaver())

    def _receive_turn(self, state: GraphState) -> GraphState:
        model = ConversationState.model_validate(state)
        model.user_goal = model.latest_user_message()
        model.active_subgraph = "clarify_subgraph"
        self._record_checkpoint(model, "receive_turn", True)
        return model.model_dump(mode="python")

    def _intent_router(self, state: GraphState) -> GraphState:
        model = ConversationState.model_validate(state)
        intent = self.classifier.classify(model)
        model.intent = intent.intent
        model.intent_confidence = intent.confidence
        model.safety_flags = list(dict.fromkeys([*model.safety_flags, f"route:{intent.route_reason}"]))
        self._record_checkpoint(model, "intent_router", True)
        return model.model_dump(mode="python")

    def _policy_gate(self, state: GraphState) -> GraphState:
        model = ConversationState.model_validate(state)
        model.active_subgraph = "clarify_subgraph" if model.intent_confidence < 0.5 else ("simple_chat_subgraph" if model.intent == "simple_chat" else "qa_subgraph")
        return model.model_dump(mode="python")

    def _execute_subgraph(self, state: GraphState) -> GraphState:
        model = ConversationState.model_validate(state)
        if model.active_subgraph == "clarify_subgraph":
            model.draft_answer = "我需要你把问题再收窄一点，比如补上文章标题、关键词、人物或命令名。"
            model.evidence = []
            model.citations = []
        elif model.active_subgraph == "simple_chat_subgraph":
            model.draft_answer = "你好，我已经准备好基于知识库帮你定位文章、引用原文并给下一步建议。"
        else:
            answer: AnswerBundle = self.rag_engine.answer(model.latest_user_message())
            model.retrieval_results = answer.retrieval_results
            model.evidence = answer.evidence
            model.draft_answer = answer.answer
            model.citations = answer.citations
            model.intent_confidence = max(model.intent_confidence, answer.confidence)
            model.safety_flags = list(dict.fromkeys([*model.safety_flags, *answer.notes]))
            self._record_checkpoint(model, "qa_subgraph", True)
        model.step_count += 1
        return model.model_dump(mode="python")

    def _response_postprocess(self, state: GraphState) -> GraphState:
        model = ConversationState.model_validate(state)
        decision = self.rollback_policy.decide(model)
        if decision.allowed:
            model = self.rollback_policy.apply(model, decision)
        model.next_actions = self.next_action_generator.generate(model)
        model.ui_blocks = [UIBlock(block_type="text", title="Answer Draft", content=model.draft_answer), *self.citation_formatter.format(model.citations), UIBlock(block_type="next_actions", title="Next Steps", content=[item.model_dump(mode="python") for item in model.next_actions])]
        return model.model_dump(mode="python")

    def _finalize(self, state: GraphState) -> GraphState:
        model = ConversationState.model_validate(state)
        model.final_response = model.draft_answer if not model.citations else f"{model.draft_answer}\n\n参考来源: {' '.join(f'[{i}] {c.doc_title}' for i, c in enumerate(model.citations, start=1))}"
        return model.model_dump(mode="python")

    @staticmethod
    def _record_checkpoint(state: ConversationState, label: str, allows_rollback: bool) -> None:
        snapshot = deepcopy(state.model_dump(mode="python"))
        snapshot["checkpoints"] = []
        checkpoint = CheckpointRecord(checkpoint_id=f"{label}-{len(state.checkpoints) + 1}", label=label, snapshot=snapshot, allows_rollback=allows_rollback)
        state.checkpoints = [*state.checkpoints, checkpoint][-8:]
        state.current_checkpoint_ref = checkpoint.checkpoint_id




