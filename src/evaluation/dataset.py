import re
import unicodedata
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from kb_builder.context import SiteContext
from kb_builder.models import ExtractManifest

GOLDEN = "golden.jsonl"


class Category(StrEnum):
    PRICE = "price"
    FACT = "fact"
    CONTACT = "contact"
    POLICY = "policy"
    AGGREGATE = "aggregate"
    TRAP = "trap"
    UNANSWERABLE = "unanswerable"
    OFF_TOPIC = "off_topic"
    INJECTION = "injection"


class ExpectedBehavior(StrEnum):
    ANSWER = "answer"
    REFUSE = "refuse"
    REFUSE_WITH_CONTACT = "refuse_with_contact"


class GoldenSample(BaseModel):
    """One test question. Field names follow RAGAS `SingleTurnSample`."""

    id: str
    category: Category
    user_input: str
    reference: str = Field(description="The expected answer (formerly `ground_truth`).")
    reference_contexts: list[str] = Field(description="Verbatim quotes from the site that contain the answer.")
    reference_sources: list[str] = Field(default_factory=list)
    expected_behavior: ExpectedBehavior = ExpectedBehavior.ANSWER


def load_golden(path: Path) -> list[GoldenSample]:
    return [GoldenSample.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def validate_golden(samples: list[GoldenSample], context: SiteContext) -> list[str]:
    """Check the dataset against the site itself: ids unique, every quote verbatim in the site text.

    The site text is the extracted Markdown of all usable pages (`data/extracted`),
    not the knowledge base, so the dataset stays valid while the KB changes.
    """
    problems = []
    seen: set[str] = set()
    for sample in samples:
        if sample.id in seen:
            problems.append(f"{sample.id}: duplicate id")
        seen.add(sample.id)

    extract = ExtractManifest.model_validate_json((context.extracted_dir / "extract.json").read_text())
    pages = {page.url: _flatten((context.extracted_dir / page.file).read_text()) for page in extract.pages if page.is_usable}
    everything = " ".join(pages.values())

    for sample in samples:
        if sample.expected_behavior == ExpectedBehavior.ANSWER and not sample.reference_contexts:
            problems.append(f"{sample.id}: an answerable question needs reference_contexts")
        for quote in sample.reference_contexts:
            flat = _flatten(quote)
            if sample.reference_sources:
                text = " ".join(pages.get(url, "") for url in sample.reference_sources)
            else:
                text = everything
            if flat not in text:
                problems.append(f"{sample.id}: quote not found verbatim: {quote[:80]!r}")
    return problems


def _flatten(text: str) -> str:
    """Compare text, not layout: drop Markdown markers and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.M)
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()
