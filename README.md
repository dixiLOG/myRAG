# myRAG

`myRAG` 是一个带教型知识图库 agent 项目骨架：

- `LangGraph` 负责 control plane：意图识别、路由、状态、回滚、澄清、引用展示、下一步提示。
- `LlamaIndex` 负责 RAG plane：Markdown 摄取、metadata、chunking、索引、检索、评测、实验。

当前仓库实现的重点是：

1. 可扩展接口先行，而不是把策略写死在 prompt 里。
2. baseline 能跑，实验能比较，状态能追踪。
3. 从一开始就把引用、回滚、评测和观测纳入骨架。

## Quick Start

```powershell
uv sync
uv run pytest
uv run myrag corpus sample
uv run myrag benchmark run
uv run uvicorn myrag.api.app:create_app --factory --reload
```

## Repository Map

- [project_spec.md](./project_spec.md)
- [control_plane_design.md](./control_plane_design.md)
- [rag_experiment_plan.md](./rag_experiment_plan.md)
- [evaluation_benchmark.md](./evaluation_benchmark.md)
- `src/myrag/`
- `tests/`

