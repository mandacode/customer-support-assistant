import asyncio
import json
import math
import os
import subprocess
from collections.abc import Coroutine
from datetime import datetime, timezone
from pathlib import Path

from evaluation.golden_dataset import GOLDEN_DATASET, GoldenDatasetSample, load_golden_dataset
from evaluation.metrics import EvaluationMetrics
from evaluation.models import EvaluationConfig, EvaluationResult, SampleResult
from evaluation.retrieval_metrics import add_retrieval_scores
from kb_builder import tracing
from kb_builder.context import SiteContext
from kb_builder.services.base_service import BaseService
from support_assistant.assistant import AssistantSettings, SupportAssistant
from support_assistant.index import Chunk


class EvaluationService(BaseService[EvaluationResult]):
    """Answer the golden-dataset questions with the assistant and grade them with RAGAS.

    Which metrics apply to which sample is decided in `EvaluationMetrics`.
    Results are saved under `evals/<site>/results/<timestamp>-<label>/` and
    attached as scores to each sample's Langfuse trace.
    """

    name = "evaluate"

    def __init__(self, config: EvaluationConfig | None = None, assistant: AssistantSettings | None = None) -> None:
        super().__init__()
        self.config = config or EvaluationConfig()
        self._assistant_settings = assistant

    def inputs(self, context: SiteContext) -> list[Path]:
        return [self._dataset_path(context), context.kb_dir]

    def run(self, context: SiteContext) -> EvaluationResult:
        samples = self._select(load_golden_dataset(self._dataset_path(context)))
        settings = self._assistant_settings or AssistantSettings(site=context.site)
        self.logger.info(
            "%d samples, assistant on %s, judge %s", len(samples), settings.site, self.config.judge_model
        )

        assistant = SupportAssistant(settings)
        results = self._answer_all(samples, assistant)
        add_retrieval_scores(results, assistant.index.chunks)
        asyncio.run(self._score_all(results))
        summary = self.summarize(results)
        out_dir = self._save(context, results, summary, settings)
        self._push_scores(results)
        return EvaluationResult(site=context.site, out_dir=out_dir, samples=results, summary=summary)

    def _dataset_path(self, context: SiteContext) -> Path:
        return self.config.evals_root / context.site / GOLDEN_DATASET

    def _select(self, samples: list[GoldenDatasetSample]) -> list[GoldenDatasetSample]:
        if self.config.sample_ids:
            wanted = set(self.config.sample_ids)
            samples = [s for s in samples if s.id in wanted]
        return samples[: self.config.limit] if self.config.limit else samples

    def _answer_all(self, samples: list[GoldenDatasetSample], assistant: SupportAssistant) -> list[SampleResult]:
        """Run the assistant on every question, each in its own trace."""
        results = []
        for index, sample in enumerate(samples, start=1):
            self.logger.info("[%d/%d] %s", index, len(samples), sample.id)
            results.append(self._answer(assistant, sample))
        return results

    @tracing.traced("eval.sample")
    def _answer(self, assistant: SupportAssistant, sample: GoldenDatasetSample) -> SampleResult:
        tracing.update(input={"id": sample.id, "question": sample.user_input}, metadata={"category": sample.category})
        reply = assistant.ask(sample.user_input)
        return SampleResult(
            id=sample.id,
            category=sample.category,
            expected_behavior=sample.expected_behavior,
            user_input=sample.user_input,
            reference=sample.reference,
            reference_contexts=sample.reference_contexts,
            response=reply.answer,
            answered=reply.answered,
            rejected_answer=reply.rejected_answer,
            retrieved_ids=[c.chunk.id for c in reply.contexts],
            retrieved_contexts=[_context_text(c.chunk) for c in reply.contexts],
            source_ids=reply.source_ids,
            trace_id=tracing.current_trace_id(),
        )

    async def _score_all(self, results: list[SampleResult]) -> None:
        """Grade every answer; fills `scores`, `reasons` and `errors` in place."""
        metrics = EvaluationMetrics(self.config)
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def grade(result: SampleResult, name: str, pending: Coroutine) -> None:
            async with semaphore:
                try:
                    metric = await asyncio.wait_for(pending, timeout=600)
                except Exception as exc:  # one failing metric must not stop the run
                    result.errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
                    result.scores[name] = None
                    return
            value = metric.value
            if name == "behavior":
                value = 1.0 if value == "pass" else 0.0
            is_missing = value is None or (isinstance(value, float) and math.isnan(value))
            result.scores[name] = None if is_missing else float(value)
            if getattr(metric, "reason", None):
                result.reasons[name] = str(metric.reason)[:1000]

        tasks = [
            grade(result, name, pending)
            for result in results
            for name, pending in metrics.for_sample(result).items()
        ]
        done = 0
        for task in asyncio.as_completed(tasks):
            await task
            done += 1
            if done % 20 == 0 or done == len(tasks):
                self.logger.info("scored %d/%d metric calls", done, len(tasks))

    @staticmethod
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

    def _save(
        self, context: SiteContext, results: list[SampleResult], summary: dict, settings: AssistantSettings
    ) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        out_dir = self.config.evals_root / context.site / "results" / f"{stamp}-{self.config.label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "samples.jsonl").open("w") as file:
            for result in results:
                file.write(result.model_dump_json() + "\n")
        document = {
            "label": self.config.label,
            "created_at": stamp,
            "git_commit": _git_commit(),
            "judge_model": self.config.judge_model,
            "assistant": {
                "site": settings.site,
                "chat_model": os.getenv("OPENAI_CHAT_MODEL"),
                "top_k": settings.top_k,
                "chunk_words": settings.chunk_words,
                "reasoning_effort": settings.reasoning_effort,
            },
            **summary,
        }
        (out_dir / "summary.json").write_text(json.dumps(document, indent=2, ensure_ascii=False))
        return out_dir

    def _push_scores(self, results: list[SampleResult]) -> None:
        """Attach every metric to the sample's Langfuse trace."""
        for result in results:
            for name, value in result.scores.items():
                if value is not None:
                    tracing.score(result.trace_id, name, value, result.reasons.get(name))


def _context_text(chunk: Chunk) -> str:
    header = " > ".join(x for x in (chunk.title, chunk.heading) if x)
    return f"{header}\n{chunk.text}"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
