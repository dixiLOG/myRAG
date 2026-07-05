# Control Plane Design

## Stable Skeleton

`receive_turn -> intent_router -> policy_gate -> execute_subgraph -> response_postprocess -> finalize`

## Baseline Subgraphs

- `qa_subgraph`
- `clarify_subgraph`
- `citation_subgraph`
- `next_step_subgraph`

## State Design

`ConversationState v1` 至少包含：

- `thread_id`
- `messages`
- `intent`
- `intent_confidence`
- `user_goal`
- `active_subgraph`
- `retrieval_results`
- `evidence`
- `draft_answer`
- `citations`
- `next_actions`
- `ui_blocks`
- `step_count`
- `checkpoint_id`
- `rollback_target`
- `last_error`
- `safety_flags`

## Rollback Rules

- 仅允许回到纯推理或纯检索节点后的 checkpoint
- 不允许穿越副作用节点回滚
- 所有 rollback 必须记录 `from_checkpoint`、`to_checkpoint` 和 `reason`

