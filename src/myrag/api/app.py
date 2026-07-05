from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI

from myrag.config import get_settings
from myrag.models import QueryRequest, QueryResponse
from myrag.runtime import Runtime, format_experiment_table, registry_summary


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(get_settings())


def create_app() -> FastAPI:
    app = FastAPI(title="myRAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        runtime = get_runtime()
        return {"status": "ok", "corpus": runtime.stats}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        runtime = get_runtime()
        return runtime.orchestrator.run(request.question, history=request.history)

    @app.get("/experiments")
    def experiments() -> dict[str, object]:
        return registry_summary()

    @app.post("/benchmark/run")
    def run_benchmark() -> dict[str, object]:
        runtime = get_runtime()
        results = runtime.experiment_runner.run(runtime.benchmark_suite)
        return {
            "results": [item.model_dump(mode="python") for item in results],
            "table": format_experiment_table(results),
        }

    return app
