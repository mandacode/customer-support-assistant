# Customer Support Assistant

A customer-support assistant for small company websites, built as a learning
project for evaluating RAG systems with [RAGAS](https://docs.ragas.io/).

It has three parts:

1. **KB Builder** (`kb`) turns a company website into a knowledge base of
   Markdown files: one per offering, plus pricing, contact, policies and so on.
2. **Support Assistant** (`support`) answers customer questions over that
   knowledge base with retrieval-augmented generation.
3. **Evaluation** (`evals`) measures the assistant on a hand-written golden
   dataset with RAGAS metrics and classic retrieval metrics (nDCG, MRR, Recall, Hit).

One rule runs through everything: **no hallucinations, only data from the site.**
Every LLM output is checked deterministically against the source text, and
anything the check cannot confirm is rejected.

Only OpenAI models are used: `gpt-5-nano` by default, `gpt-5-mini` where nano
is not good enough, and `text-embedding-3-small` for embeddings.

## Setup

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mandacode/customer-support-assistant.git
cd customer-support-assistant
uv sync
cp .env.example .env
```

Edit `.env`:

| Variable | Required | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI API key |
| `KB_SITE` | for `support` | site folder in `knowledge_base/`, e.g. `art-mat.com.pl` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | no | tracing in [Langfuse](https://langfuse.com/); off when empty |
| `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_REASONING_EFFORT` | no | override the pinned defaults in `src/kb_builder/llm.py` |

Every LLM response is cached in `data/llm_cache/`, and RAGAS judge calls in
`data/ragas_cache/`. A repeated run with unchanged input costs nothing and
gives the same result.

## 1. Build a knowledge base

```bash
uv run kb build https://art-mat.com.pl/
```

The pipeline runs these stages and writes the result to `knowledge_base/<site>/`:

| Stage | LLM | What it does |
|---|---|---|
| `crawl` | – | sitemap plus internal links, `robots.txt`; caches raw HTML and PDFs in `data/raw/` |
| `extract` | – | clean Markdown; drops navigation, footers and mobile duplicates; flags CMS junk, login and thin pages |
| `segment` | – | splits pages into sections by headings |
| `profile` | nano | company facts and a support checklist; offerings are chosen by 5 votes from phrases found on the site |
| `classify` | nano | gives every section a topic and offerings, picked from fixed lists |
| `merge` | – | groups sections into future files and removes duplicates (number-aware) |
| `normalize` | nano / mini | tidies each file; the output is verified, otherwise the source text is kept |

`normalize` rejects an LLM rewrite when it:

- adds a number or an e-mail address the source does not have,
- moves a number out of its source context (for example a price into another row),
- uses words that are not in the source (below 90% coverage),
- drops a substantial source sentence.

A rejected rewrite is retried with `gpt-5-mini`; if that also fails, the file
is written from the source text unchanged.

Useful options:

```bash
uv run kb build https://art-mat.com.pl/ --from classify   # rerun from a stage, earlier stages from cache
uv run kb candidates art-mat.com.pl                      # offering candidates found on the site (no LLM)
uv run kb crawl --help                                   # every stage is also its own command
```

Building art-mat.com.pl from scratch costs about $0.03–0.07 and takes a few minutes.

## 2. Ask the assistant

```bash
uv run support chat                                      # interactive
uv run support ask "Ile kosztuje wyważenie felg 17 cali?" -c   # one question, -c shows retrieved chunks
```

The assistant retrieves the 4 most similar chunks (cosine similarity over
`text-embedding-3-small` embeddings, kept in `data/index/<site>/index.json`)
and answers with `gpt-5-nano`. It:

- answers only from the retrieved context, in the customer's language;
- cites sources, and can only cite chunks it was given;
- says it does not know, and points to the contact details, when the context has no answer;
- has its answer replaced by a safe fallback if the answer contains a number or
  an e-mail address that is not in the context.

## 3. Evaluate

### Golden dataset

`evals/<site>/golden_dataset.jsonl` holds the test questions, one JSON object
per line. Field names follow RAGAS `SingleTurnSample`:

| Field | Meaning |
|---|---|
| `id`, `category` | stable id; `price`, `fact`, `contact`, `policy`, `aggregate`, `trap`, `unanswerable`, `off_topic`, `injection` |
| `user_input` | the customer's question |
| `reference` | the expected answer (formerly `ground_truth`) |
| `reference_contexts` | verbatim quotes from the site that contain the answer |
| `expected_behavior` | `answer`, `refuse` or `refuse_with_contact` |

The quotes are taken from the site, not from the knowledge base, so the
dataset stays valid while the KB changes. Check that every quote really is on
the site:

```bash
uv run evals validate art-mat.com.pl
```

### Run an evaluation

```bash
.venv/bin/evals run art-mat.com.pl --label baseline              # all questions
.venv/bin/evals run art-mat.com.pl --label smoke --limit 5       # cheap smoke test
.venv/bin/evals run art-mat.com.pl --ids gap-hours,agg-services  # selected questions
.venv/bin/evals retrieval art-mat.com.pl 20261019-1334-baseline  # recompute retrieval metrics only (free)
```

(`.venv/bin/evals` instead of `uv run evals`: `uv run` occasionally hung on long runs.)

Each run is saved in `evals/<site>/results/<timestamp>-<label>/`:

- `summary.json`: mean of every metric, overall and per category, plus the git commit and assistant settings;
- `samples.jsonl`: per question, the response, retrieved chunks, scores and the judge's reasons.

With Langfuse configured, every question is a trace (`eval.sample` → retrieval
→ OpenAI calls) with all metrics attached as scores.

### Metrics

| Metric | Computed by | Applies to | Measures |
|---|---|---|---|
| `behavior` | judge (custom `DiscreteMetric`) | all | answered, or refused, as expected |
| `faithfulness` | RAGAS | responses that answer | every claim is supported by the retrieved context (hallucinations) |
| `answer_relevancy` | RAGAS | `answer` questions | the response addresses the question (useful only to compare runs) |
| `context_precision` | RAGAS | `answer` questions | relevant chunks are ranked high |
| `context_recall` | RAGAS | `answer` questions | the retrieved chunks cover the reference |
| `factual_correctness` | RAGAS (recall mode) | `answer` questions | the response contains the reference's facts |
| `hit_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k` | code, no LLM | `answer` questions | classic retrieval metrics; a chunk is relevant if it contains a reference quote |

The judge is `gpt-5-mini`, a stronger model than the `gpt-5-nano` it grades.
A full run on 50 questions costs about $1–2 without cache. With cache, only
questions whose answer changed are graded again, plus `behavior`, which cannot
be cached.

Metrics such as `answer_relevancy` and `context_recall` would punish a correct
refusal, so they are not computed for questions the assistant should refuse.

## Example: art-mat.com.pl

[art-mat.com.pl](https://art-mat.com.pl/) is a tyre service in Stalowa Wola: a
one-page WordPress site with a price list and two policy PDFs. The knowledge
base built from it has 11 files: 6 offerings, pricing, contact, a process
note and 2 policies. The golden dataset has 50 questions.

Results of the current baseline (`evals/art-mat.com.pl/results/20261019-1555-baseline-v2/`,
assistant: `gpt-5-nano`, `top_k=4`; judge: `gpt-5-mini`):

| | n | `behavior` | `faithfulness` | `factual_correctness` | `context_recall` | `recall_at_k` | `ndcg_at_k` |
|---|---|---|---|---|---|---|---|
| **overall** | 50 | 0.82 | 0.923 | 0.59 | 0.895 | 0.914 | 0.811 |
| aggregate | 3 | 0.667 | 1.0 | 0.5 | 0.762 | 0.8 | 0.682 |
| contact | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.591 |
| fact | 10 | 0.9 | 0.98 | 0.466 | 0.9 | 0.9 | 0.88 |
| injection | 3 | 1.0 | – | – | – | – | – |
| off_topic | 2 | 0.5 | 1.0 | – | – | – | – |
| policy | 6 | 0.833 | 0.9 | 0.667 | 0.833 | 0.833 | 0.772 |
| price | 11 | 0.909 | 0.85 | 0.652 | 1.0 | 1.0 | 0.89 |
| trap | 3 | 0.0 | 1.0 | 0.3 | 0.639 | 0.833 | 0.721 |
| unanswerable | 9 | 0.889 | 0.5 | – | – | – | – |

"–": the metric does not apply to that category (see [Metrics](#metrics)).

What the numbers show:

- **Prices, contact and single facts work**: the right chunk is found (`recall_at_k` 1.0 for prices
  and contact) and the answer sticks to it (`faithfulness` 0.92 overall).
- **Retrieval misses facts buried in long paragraphs** (the company's NIP, "we also sell rims") and
  finds only 2 of the 5 reference quotes for "What services do you offer?" (`top_k=4` is too small for it).
- **Traps fail** (`behavior` 0.0): the assistant repeats "od 50 PLN" instead of noticing that the
  cheapest item in the price table costs 100 PLN.
- **The number guard is too strict**: a correct "60 PLN difference" was replaced by the fallback,
  because 60 itself is not in the context.
- **Refusals mostly work** (`behavior` 0.89 for unanswerable questions), but the assistant still
  recommends tyres instead of declining when asked for advice the knowledge base does not have.

The first baseline (`20261019-1334-baseline`) used a stricter evaluation that penalised correct
refusals; it is kept for reference. Every RAG change is measured against the current baseline
with a new labelled run.

## Project layout

```
src/
  kb_builder/          # `kb`: website -> knowledge base
    services/          # one *_service.py per pipeline stage
    pipeline.py        # runs the stages in order (--from / --to)
    grounding.py       # deterministic checks against the source text
    llm.py             # OpenAI client with structured output and disk cache
    prompts.py
  support_assistant/   # `support`: RAG assistant
  evaluation/          # `evals`: golden dataset, RAGAS, retrieval metrics
evals/<site>/          # golden dataset and evaluation results (versioned)
knowledge_base/<site>/ # generated knowledge base (not versioned)
data/                  # caches and intermediate stage outputs (not versioned)
```

## License

MIT
