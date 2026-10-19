import asyncio
import json
import logging
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from evaluation.golden_dataset import ExpectedBehavior, GoldenDatasetSample
from evaluation.prompts import BEHAVIOR_PROMPT
from kb_builder import tracing
from support_assistant.assistant import AssistantSettings, SupportAssistant

logger = logging.getLogger("evaluation.runner")

# The judge should be stronger than the assistant it grades (gpt-5-nano).
JUDGE_MODEL = "gpt-5-mini-2025-08-07"


class EvalConfig(BaseModel):
    judge_model: str = JUDGE_MODEL
    # Reasoning judges spend output tokens on thinking; RAGAS's default 1024 is too few.
    judge_max_tokens: int = 16000
    concurrency: int = Field(4, ge=1)
    label: str = "run"
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


def answer_all(samples: list[GoldenDatasetSample], settings: AssistantSettings) -> list[SampleResult]:
    """Run the assistant on every question, each in its own trace."""
    assistant = SupportAssistant(settings)
    results = []
    for index, sample in enumerate(samples, start=1):
        logger.info("[%d/%d] %s", index, len(samples), sample.id)
        results.append(_answer(assistant, sample))
    return results


@tracing.traced("eval.sample")
def _answer(assistant: SupportAssistant, sample: GoldenDatasetSample) -> SampleResult:
    tracing.update(input={"id": sample.id, "question": sample.user_input}, metadata={"category": sample.category})
    reply = assistant.ask(sample.user_input)
    return SampleResult(
        id=sample.id,
        category=sample.category,
        expected_behavior=sample.expected_behavior,
        user_input=sample.user_input,
        reference=sample.reference,
        response=reply.answer,
        answered=reply.answered,
        rejected_answer=reply.rejected_answer,
        retrieved_ids=[c.chunk.id for c in reply.contexts],
        retrieved_contexts=[_context_text(c.chunk) for c in reply.contexts],
        source_ids=reply.source_ids,
        trace_id=tracing.current_trace_id(),
    )


async def score_all(results: list[SampleResult], config: EvalConfig) -> None:
    """Grade every answer with RAGAS metrics; fills `scores`, `reasons` and `errors` in place.

    Every sample gets `behavior` (did it answer / refuse as expected) and
    `faithfulness` (no claims beyond the retrieved context). Only questions that
    should be answered also get the reference-based metrics: for a correct
    refusal, answer relevancy and context recall would punish the right outcome.
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics import DiscreteMetric
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
        FactualCorrectness,
    )

    from ragas.cache import DiskCacheBackend

    client = AsyncOpenAI(timeout=180, max_retries=2)
    # Judge calls are cached on disk: re-grading an unchanged answer costs nothing,
    # only answers that changed since the last run are sent to the judge again.
    llm = llm_factory(
        config.judge_model,
        client=client,
        max_tokens=config.judge_max_tokens,
        cache=DiskCacheBackend(cache_dir=str(config.cache_dir)),
    )
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    behavior = DiscreteMetric(name="behavior", allowed_values=["pass", "fail"], prompt=BEHAVIOR_PROMPT)
    faithfulness = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecisionWithReference(llm=llm)
    recall = ContextRecall(llm=llm)
    correctness = FactualCorrectness(llm=llm)

    semaphore = asyncio.Semaphore(config.concurrency)

    async def run(result: SampleResult, name: str, coro) -> None:
        async with semaphore:
            try:
                metric = await asyncio.wait_for(coro, timeout=600)
            except Exception as exc:  # one failing metric must not stop the run
                result.errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
                result.scores[name] = None
                return
        value = metric.value
        if name == "behavior":
            value = 1.0 if value == "pass" else 0.0
        result.scores[name] = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
        if getattr(metric, "reason", None):
            result.reasons[name] = str(metric.reason)[:1000]

    tasks = []
    for result in results:
        tasks.append(
            run(
                result,
                "behavior",
                behavior.ascore(
                    llm=llm,
                    expected_behavior=result.expected_behavior,
                    reference=result.reference,
                    user_input=result.user_input,
                    response=result.response,
                ),
            )
        )
        tasks.append(
            run(
                result,
                "faithfulness",
                faithfulness.ascore(
                    user_input=result.user_input, response=result.response, retrieved_contexts=result.retrieved_contexts
                ),
            )
        )
        if result.expected_behavior != ExpectedBehavior.ANSWER:
            continue
        tasks += [
            run(result, "answer_relevancy", relevancy.ascore(user_input=result.user_input, response=result.response)),
            run(
                result,
                "context_precision",
                precision.ascore(
                    user_input=result.user_input, reference=result.reference, retrieved_contexts=result.retrieved_contexts
                ),
            ),
            run(
                result,
                "context_recall",
                recall.ascore(
                    user_input=result.user_input, retrieved_contexts=result.retrieved_contexts, reference=result.reference
                ),
            ),
            run(result, "factual_correctness", correctness.ascore(response=result.response, reference=result.reference)),
        ]

    done = 0
    for task in asyncio.as_completed(tasks):
        await task
        done += 1
        if done % 20 == 0 or done == len(tasks):
            logger.info("scored %d/%d metric calls", done, len(tasks))


def summarize(results: list[SampleResult]) -> dict:
    """Mean of every metric, overall and per category (missing scores are skipped)."""

    def means(items: list[SampleResult]) -> dict[str, float | int]:
        names = sorted({name for r in items for name in r.scores})
        out: dict[str, float | int] = {"samples": len(items)}
        for name in names:
            values = [r.scores[name] for r in items if r.scores.get(name) is not None]
            if values:
                out[name] = round(sum(values) / len(values), 3)
        return out

    categories = sorted({r.category for r in results})
    return {
        "overall": means(results),
        "by_category": {c: means([r for r in results if r.category == c]) for c in categories},
        "errors": sum(len(r.errors) for r in results),
    }


def save_run(
    out_root: Path, results: list[SampleResult], config: EvalConfig, settings: AssistantSettings
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out_dir = out_root / f"{stamp}-{config.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "samples.jsonl").open("w") as file:
        for result in results:
            file.write(result.model_dump_json() + "\n")
    summary = {
        "label": config.label,
        "created_at": stamp,
        "git_commit": _git_commit(),
        "judge_model": config.judge_model,
        "assistant": {
            "site": settings.site,
            "chat_model": os.getenv("OPENAI_CHAT_MODEL"),
            "top_k": settings.top_k,
            "chunk_words": settings.chunk_words,
            "reasoning_effort": settings.reasoning_effort,
        },
        **summarize(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out_dir


def push_scores(results: list[SampleResult]) -> None:
    """Attach every metric to the sample's Langfuse trace."""
    for result in results:
        for name, value in result.scores.items():
            if value is not None:
                tracing.score(result.trace_id, name, value, result.reasons.get(name))


def _context_text(chunk) -> str:
    header = " > ".join(x for x in (chunk.title, chunk.heading) if x)
    return f"{header}\n{chunk.text}"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
