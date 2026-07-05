from __future__ import annotations

import json

import typer

from myrag.config import get_settings
from myrag.corpus import corpus_stats, load_markdown_corpus
from myrag.rag import ExperimentRunner
from myrag.runtime import Runtime, format_experiment_table, load_benchmark_suite, registry_summary

app = typer.Typer(help="myRAG CLI")
corpus_app = typer.Typer(help="Corpus tools")
benchmark_app = typer.Typer(help="Benchmark tools")
agent_app = typer.Typer(help="Agent tools")
app.add_typer(corpus_app, name="corpus")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(agent_app, name="agent")


@corpus_app.command("sample")
def corpus_sample() -> None:
    settings = get_settings()
    documents = load_markdown_corpus(settings.sample_corpus_dir)
    typer.echo(json.dumps(corpus_stats(documents), ensure_ascii=False, indent=2))


@benchmark_app.command("registry")
def benchmark_registry() -> None:
    typer.echo(json.dumps(registry_summary(), ensure_ascii=False, indent=2))


@benchmark_app.command("run")
def benchmark_run() -> None:
    settings = get_settings()
    documents = load_markdown_corpus(settings.sample_corpus_dir)
    benchmark_suite = load_benchmark_suite(settings.benchmark_path)
    results = ExperimentRunner(documents, settings.artifacts_dir).run(benchmark_suite)
    typer.echo(format_experiment_table(results))


@agent_app.command("ask")
def agent_ask(question: str) -> None:
    runtime = Runtime(get_settings())
    response = runtime.orchestrator.run(question)
    typer.echo(response.answer)
    typer.echo(json.dumps(response.model_dump(mode="python"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
