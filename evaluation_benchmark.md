# Evaluation Benchmark

## Benchmark Dimensions

### Content Type

- 随笔
- 教程
- 长文
- 读书笔记

### Query Type

- 事实查找
- 观点定位
- 步骤总结
- 跨文档关联

## Metrics

### Retrieval

- `hit_rate`
- `MRR`
- `nDCG`
- `empty_retrieval_rate`

### Selection / Ranking

- `top1_hit_rate`
- `evidence_coverage`

### Answer

- `faithfulness`
- `citation_completeness`
- `latency_ms`
- `token_cost`

## Required Outputs

- `run_id`
- `corpus_version`
- `index_version`
- `embedding_version`
- `retrieval_version`
- `rerank_version`
- `metrics`
- `cost`
- `latency`
