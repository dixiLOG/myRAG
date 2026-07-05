from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MYRAG_", env_file=".env", extra="ignore")

    workspace_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    artifacts_dir: Path | None = None
    sample_corpus_dir: Path | None = None
    benchmark_path: Path | None = None
    default_top_k: int = 5
    checkpoint_limit: int = 8

    def model_post_init(self, __context: object) -> None:
        if self.artifacts_dir is None:
            self.artifacts_dir = self.workspace_root / "artifacts"
        if self.sample_corpus_dir is None:
            self.sample_corpus_dir = self.workspace_root / "examples" / "sample_corpus"
        if self.benchmark_path is None:
            self.benchmark_path = self.workspace_root / "benchmarks" / "sample_benchmark.yaml"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
