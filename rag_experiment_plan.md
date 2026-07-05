# RAG Experiment Plan

## Baseline

`Markdown -> metadata normalization -> structured chunking -> dense retrieval -> cited answer`

## RetrievalPipelineSpec

- `embedding_model`
- `chunk_strategy`
- `metadata_policy`
- `retriever_type`
- `retriever_params`
- `reranker_type`
- `reranker_params`
- `response_mode`

## Experiment Phases

1. Chunking A vs B
2. Embedding A vs B
3. Dense vs hybrid
4. Rerank off vs on

## Decision Rule

仅当质量有明确提升，且延迟/成本仍可接受时，新策略才进入候选默认链路。

