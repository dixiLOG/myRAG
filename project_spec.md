# Project Spec

## Objective

构建一个本地单用户知识图库 agent 骨架，用真实 Markdown 博客语料学习：

- 如何用 `LlamaIndex` 搭一个可实验、可评测的 RAG plane
- 如何用 `LangGraph` 搭一个可扩展、可回滚的 control plane

## V1 Scope

- 单用户、本地运行
- 数据源仅支持 Markdown 语料目录
- 基线支持：
  - 语料摄取
  - metadata 规范化
  - 结构化 chunking
  - baseline dense retrieval
  - 实验 registry
  - retrieval benchmark
  - control plane graph
  - citations / next actions / rollback skeleton

## Success Criteria

- 能构建本地语料索引并检索来源片段
- 能执行最小 LangGraph 工作流
- 所有回答型输出都能附带结构化引用
- 实验配置可登记、可执行、可比较
- rollback policy 和 checkpoint contract 明确

## Commands

```powershell
uv sync
uv run pytest
uv run myrag corpus sample
uv run myrag benchmark run
uv run uvicorn myrag.api.app:create_app --factory --reload
```

## Boundaries

- Always:
  - 只在结构化 state 中传递控制面数据
  - 所有检索结果都带 metadata 和 source id
  - 重要节点都可生成 checkpoint
- Ask first:
  - 引入外部私有数据源
  - 引入有副作用的写操作工具
  - 改成多用户或 SaaS 部署
- Never:
  - 自动抓网页后直接信任入库
  - 自动执行 shell / arbitrary code
  - 在没有引用的情况下输出“确定性知识答案”

