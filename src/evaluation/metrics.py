from collections.abc import Coroutine
from typing import Any

from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from evaluation.golden_dataset import ExpectedBehavior
from evaluation.models import EvaluationConfig, SampleResult
from evaluation.prompts import BEHAVIOR_PROMPT

EMBEDDING_MODEL = "text-embedding-3-small"


class EvaluationMetrics:
    """The RAGAS judge and every metric, and which of them apply to a sample.

    - `behavior` (did it answer / refuse as expected): every sample.
    - `faithfulness` (no claims beyond the retrieved context): only responses that
      answer; a refusal ("I don't have this information") has nothing to check,
      and RAGAS would score it as unsupported. Invented facts inside a refusal
      fail `behavior`.
    - reference-based metrics: only questions that should be answered; for a
      correct refusal, answer relevancy and context recall would punish the
      right outcome.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        client = AsyncOpenAI(timeout=180, max_retries=2)
        # Judge calls are cached on disk: re-grading an unchanged answer costs nothing,
        # only answers that changed since the last run are sent to the judge again.
        self.llm = llm_factory(
            config.judge_model,
            client=client,
            max_tokens=config.judge_max_tokens,
            cache=DiskCacheBackend(cache_dir=str(config.cache_dir)),
        )
        # DiscreteMetric builds its response model at runtime and the disk cache
        # cannot pickle it, so `behavior` (one call per sample) runs uncached.
        self.uncached_llm = llm_factory(config.judge_model, client=client, max_tokens=config.judge_max_tokens)
        embeddings = embedding_factory("openai", model=EMBEDDING_MODEL, client=client)

        self.behavior = DiscreteMetric(name="behavior", allowed_values=["pass", "fail"], prompt=BEHAVIOR_PROMPT)
        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=embeddings)
        self.context_precision = ContextPrecisionWithReference(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)
        # Recall: does the response contain the reference's facts. F1 would also punish
        # correct extra details; invented ones are `faithfulness`'s job.
        self.factual_correctness = FactualCorrectness(llm=self.llm, mode="recall")

    def for_sample(self, result: SampleResult) -> dict[str, Coroutine[Any, Any, Any]]:
        """Metric name -> pending score for every metric that applies to this sample."""
        question, response, contexts, reference = (
            result.user_input,
            result.response,
            result.retrieved_contexts,
            result.reference,
        )
        scores = {
            "behavior": self.behavior.ascore(
                llm=self.uncached_llm,
                expected_behavior=result.expected_behavior,
                reference=reference,
                user_input=question,
                response=response,
            ),
        }
        if result.answered:
            scores["faithfulness"] = self.faithfulness.ascore(
                user_input=question, response=response, retrieved_contexts=contexts
            )
        if result.expected_behavior == ExpectedBehavior.ANSWER:
            scores |= {
                "answer_relevancy": self.answer_relevancy.ascore(user_input=question, response=response),
                "context_precision": self.context_precision.ascore(
                    user_input=question, reference=reference, retrieved_contexts=contexts
                ),
                "context_recall": self.context_recall.ascore(
                    user_input=question, retrieved_contexts=contexts, reference=reference
                ),
                "factual_correctness": self.factual_correctness.ascore(response=response, reference=reference),
            }
        return scores
