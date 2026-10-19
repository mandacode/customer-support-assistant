from pathlib import Path

from pydantic import BaseModel, Field

from kb_builder.models import StageResult

# The judge should be stronger than the assistant it grades (gpt-5-nano).
JUDGE_MODEL = "gpt-5-mini-2025-08-07"


class EvaluationConfig(BaseModel):
    label: str = "run"
    evals_root: Path = Path("evals")
    # Only these sample ids (all when empty), then only the first `limit` of them.
    sample_ids: list[str] = Field(default_factory=list)
    limit: int | None = Field(None, ge=1)
    judge_model: str = JUDGE_MODEL
    # Reasoning judges spend output tokens on thinking; RAGAS's default 1024 is too few.
    judge_max_tokens: int = 16000
    concurrency: int = Field(4, ge=1)
    cache_dir: Path = Path("data/ragas_cache")


class SampleResult(BaseModel):
    id: str
    category: str
    expected_behavior: str
    user_input: str
    reference: str
    response: str
    answered: bool
    rejected_answer: str | None
    retrieved_ids: list[str]
    retrieved_contexts: list[str]
    source_ids: list[str]
    trace_id: str | None
    scores: dict[str, float | None] = Field(default_factory=dict)
    reasons: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class EvaluationResult(StageResult):
    samples: list[SampleResult]
    summary: dict

    def counts(self) -> dict[str, int]:
        return {"samples": len(self.samples), "errors": self.summary.get("errors", 0)}
