import math

from evaluation.golden_dataset import ExpectedBehavior
from evaluation.models import SampleResult
from kb_builder.grounding import numbers, stems, ungrounded_words
from support_assistant.index import Chunk

# Share of a quote's meaningful words that a chunk must contain to count as a match.
# Below 1.0 because the knowledge base is a tidied copy of the site (tables, wording).
QUOTE_WORD_COVERAGE = 0.8


def chunk_text(chunk: Chunk) -> str:
    """Title, heading and text: a quote may span a heading and the line under it."""
    return "\n".join(x for x in (chunk.title, chunk.heading, chunk.text) if x)


def contains_quote(text: str, quote: str) -> bool:
    """True if the text holds the quote: all its numbers and most of its words."""
    quote_numbers = set(numbers(quote))
    if quote_numbers and not quote_numbers <= set(numbers(text)):
        return False
    words = stems(quote)
    if not words:
        return False
    coverage = 1 - len(ungrounded_words(quote, text)) / len(words)
    return coverage >= QUOTE_WORD_COVERAGE


def retrieval_scores(retrieved_ids: list[str], all_chunks: list[Chunk], quotes: list[str]) -> dict[str, float]:
    """Classic IR metrics of one ranking, with relevance labels taken from the reference quotes.

    A chunk is relevant if it contains at least one reference quote. With k the
    number of retrieved chunks:
    - hit_at_k: 1 if any of the top k is relevant;
    - recall_at_k: share of the reference quotes found in the top k;
    - mrr: 1 / rank of the first relevant chunk (0 if none);
    - ndcg_at_k: binary-relevance nDCG, the ideal ranking built from every
      relevant chunk in the whole index.
    """
    by_id = {c.id: c for c in all_chunks}
    texts = {c.id: chunk_text(c) for c in all_chunks}
    covered = {cid: {q for q in quotes if contains_quote(texts[cid], q)} for cid in texts}

    ranking = [cid for cid in retrieved_ids if cid in by_id]
    relevant = [bool(covered[cid]) for cid in ranking]
    k = len(ranking)

    found = set().union(*(covered[cid] for cid in ranking)) if ranking else set()
    first = next((rank for rank, is_relevant in enumerate(relevant, start=1) if is_relevant), None)

    dcg = sum(1 / math.log2(rank + 1) for rank, is_relevant in enumerate(relevant, start=1) if is_relevant)
    relevant_in_index = sum(1 for cid in covered if covered[cid])
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, relevant_in_index) + 1))

    return {
        "hit_at_k": 1.0 if any(relevant) else 0.0,
        "recall_at_k": len(found) / len(quotes),
        "mrr": 1 / first if first else 0.0,
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
    }


def add_retrieval_scores(results: list[SampleResult], all_chunks: list[Chunk]) -> None:
    """Fill retrieval metrics for questions that should be answered and have reference quotes."""
    for result in results:
        if result.expected_behavior != ExpectedBehavior.ANSWER or not result.reference_contexts:
            continue
        result.scores |= retrieval_scores(result.retrieved_ids, all_chunks, result.reference_contexts)
