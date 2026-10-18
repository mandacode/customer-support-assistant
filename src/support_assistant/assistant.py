import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

from kb_builder import tracing
from kb_builder.grounding import emails, numbers
from kb_builder.llm import LLMClient
from support_assistant.index import Chunk, KnowledgeIndex

NO_SOURCE = "none"

SYSTEM_PROMPT = """\
You are the customer-support assistant of the company whose website is {site}.
Answer the customer's question using ONLY the CONTEXT: excerpts from the company's
knowledge base, each with an id.

Rules:
- Use only facts from the CONTEXT. Never guess, never add general knowledge about the industry.
- Copy prices, numbers, names, addresses and contact details exactly as written.
- If the CONTEXT does not answer the question, set answered to false and say briefly that you
  do not have this information. If the CONTEXT contains contact details, point the customer to them.
- If the CONTEXT contains contradicting information, say so instead of choosing one.
- Answer in the language of the question, briefly: a few sentences or a short list.
- source_ids: the ids of the excerpts your answer is based on.
"""


class AssistantSettings(BaseModel):
    site: str
    kb_root: Path = Path("knowledge_base")
    data_root: Path = Path("data")
    top_k: int = Field(4, ge=1)
    chunk_words: int = Field(300, ge=50)
    reasoning_effort: str = "low"
    # Returned instead of an answer that states numbers or e-mails absent from the context.
    fallback_answer: str = "Nie mam pewnej informacji na ten temat. Skontaktuj się proszę bezpośrednio z firmą."

    @classmethod
    def from_env(cls) -> "AssistantSettings":
        load_dotenv()
        site = os.getenv("KB_SITE")
        if not site:
            raise RuntimeError("Set KB_SITE in .env to the site folder in knowledge_base/, e.g. KB_SITE=art-mat.com.pl")
        return cls(site=site, top_k=int(os.getenv("ASSISTANT_TOP_K", "4")))

    @property
    def kb_dir(self) -> Path:
        return self.kb_root / self.site

    @property
    def index_dir(self) -> Path:
        return self.data_root / "index" / self.site


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class AssistantAnswer(BaseModel):
    question: str
    answer: str
    answered: bool
    source_ids: list[str]
    contexts: list[RetrievedChunk]
    # Set when the model's answer was replaced because it was not supported by the context.
    rejected_answer: str | None = None
    problems: list[str] = Field(default_factory=list)


class SupportAssistant:
    """Retrieval-augmented customer-support assistant over one site's knowledge base."""

    def __init__(self, settings: AssistantSettings | None = None, llm: LLMClient | None = None) -> None:
        self.settings = settings or AssistantSettings.from_env()
        self.llm = llm or LLMClient()
        self._index: KnowledgeIndex | None = None

    @property
    def index(self) -> KnowledgeIndex:
        if self._index is None:
            self._index = KnowledgeIndex.load_or_build(
                self.settings.kb_dir, self.settings.index_dir, self.llm, self.settings.chunk_words
            )
        return self._index

    @tracing.traced("support.ask")
    def ask(self, question: str) -> AssistantAnswer:
        tracing.update(input={"question": question}, metadata={"site": self.settings.site})
        contexts = self.retrieve(question)
        ids = [c.chunk.id for c in contexts]

        schema = _answer_schema(ids)
        reply = self.llm.parse(
            system=SYSTEM_PROMPT.format(site=self.settings.site),
            user=_prompt(question, contexts),
            schema=schema,
            reasoning_effort=self.settings.reasoning_effort,
        )
        result = AssistantAnswer(
            question=question,
            answer=reply.answer.strip(),
            answered=reply.answered,
            source_ids=[i for i in dict.fromkeys(reply.source_ids) if i != NO_SOURCE],
            contexts=contexts,
        )
        self._guard(result)
        tracing.update(
            output={"answer": result.answer, "answered": result.answered, "source_ids": result.source_ids},
            metadata={"problems": result.problems},
        )
        return result

    @tracing.traced("support.retrieve")
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        [vector] = self.llm.embed([question])
        found = [RetrievedChunk(chunk=chunk, score=score) for chunk, score in self.index.search(vector, self.settings.top_k)]
        tracing.update(output=[{"id": r.chunk.id, "score": round(r.score, 3)} for r in found])
        return found

    def _guard(self, result: AssistantAnswer) -> None:
        """Replace an answer whose numbers or e-mails do not occur in the retrieved context."""
        context = "\n".join(c.chunk.text for c in result.contexts)
        known_numbers = set(numbers(context))
        added_numbers = sorted({n for n in numbers(result.answer) if n not in known_numbers})
        added_emails = sorted(emails(result.answer) - emails(context))
        if added_numbers:
            result.problems.append(f"numbers not in context: {', '.join(added_numbers)}")
        if added_emails:
            result.problems.append(f"e-mails not in context: {', '.join(added_emails)}")
        if result.problems:
            result.rejected_answer = result.answer
            result.answer = self.settings.fallback_answer
            result.answered = False


def _answer_schema(chunk_ids: list[str]) -> type[BaseModel]:
    """The model can only cite chunks it was given."""
    source_id = Literal[tuple(chunk_ids or [NO_SOURCE])]  # type: ignore[valid-type]
    return create_model(
        "SupportReply",
        answer=(str, ...),
        answered=(bool, Field(description="False when the context does not contain the answer.")),
        source_ids=(list[source_id], ...),
    )


def _prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    lines = ["CONTEXT:"]
    for item in contexts:
        chunk = item.chunk
        header = " > ".join(x for x in (chunk.title, chunk.heading) if x)
        lines += [f"--- id: {chunk.id}", f"{header}", chunk.text, ""]
    lines += ["QUESTION:", question]
    return "\n".join(lines)
