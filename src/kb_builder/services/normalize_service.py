import json
from pathlib import Path

from kb_builder.context import SiteContext
from kb_builder.grounding import emails, numbers, stems, ungrounded_words
from kb_builder.llm import LLMClient
from kb_builder.models import (
    ClassifiedSection,
    DocumentGroup,
    DocumentReport,
    NormalizeConfig,
    NormalizedDocument,
    NormalizeResult,
    SiteProfile,
)
from kb_builder.services.base_service import BaseService
from kb_builder.services.classify_service import CLASSIFIED, read_classified
from kb_builder.services.merge_service import DOCUMENTS
from kb_builder.services.profile_service import PROFILE

REPORT = "normalize.json"

SYSTEM_PROMPT = """\
You turn raw text scraped from a company website into one clean knowledge-base document
for a customer-support assistant.

Hard rules:
- Use only information in SOURCE. Do not add facts, examples, advice, greetings or marketing.
- Keep every fact: names, numbers, prices, dimensions, units, contact details and conditions,
  copied exactly with the same digits and spelling.
- Remove website leftovers: buttons, navigation, calls to action ("Wyznacz trasę", "Zobacz ofertę"),
  calculator widgets, empty placeholders and repeated text.
- Write in the site's language with the source's own wording. You may shorten or reorder,
  but do not paraphrase with new words.
- If the source contains contradicting values, keep both; do not resolve them.

Structure:
- `##` headings for the parts of the document, bullet lists for features or options.
- Markdown tables for price lists and specifications: one row per item, columns from the source.
"""


class NormalizeService(BaseService[NormalizeResult]):
    """Write the knowledge-base Markdown files, verifying every LLM rewrite.

    The LLM only tidies the text of each document group. Its output is accepted
    only if every number and e-mail address occurs in the source and almost all
    of its words do; otherwise the file is written from the source text as is.
    """

    name = "normalize"

    def __init__(self, config: NormalizeConfig | None = None, llm: LLMClient | None = None) -> None:
        super().__init__()
        self.config = config or NormalizeConfig()
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def inputs(self, context: SiteContext) -> list[Path]:
        return [
            context.merged_dir / DOCUMENTS,
            context.classified_dir / CLASSIFIED,
            context.profile_dir / PROFILE,
        ]

    def run(self, context: SiteContext) -> NormalizeResult:
        merged = json.loads((context.merged_dir / DOCUMENTS).read_text())
        documents = [DocumentGroup.model_validate(d) for d in merged["documents"]]
        sections = {s.id: s for s in read_classified(context.classified_dir / CLASSIFIED)}
        profile = SiteProfile.model_validate_json((context.profile_dir / PROFILE).read_text())

        self._remove_previous(context)
        reports: list[DocumentReport] = []
        for document in documents:
            members = [sections[sid] for sid in document.section_ids]
            report = self._write_document(context, document, members, profile)
            reports.append(report)
            status = "ok" if report.normalized else "FALLBACK"
            self.logger.info("  %-8s %-45s %s", status, document.id, "; ".join(report.problems))

        context.normalized_dir.mkdir(parents=True, exist_ok=True)
        (context.normalized_dir / REPORT).write_text(
            json.dumps([r.model_dump() for r in reports], indent=2, ensure_ascii=False)
        )
        return NormalizeResult(site=context.site, out_dir=context.kb_dir, reports=reports)

    def _write_document(
        self,
        context: SiteContext,
        document: DocumentGroup,
        members: list[ClassifiedSection],
        profile: SiteProfile,
    ) -> DocumentReport:
        source = "\n\n".join(section.text for section in members)
        output = self.llm.parse(
            system=SYSTEM_PROMPT,
            user=self._prompt(document, profile, source),
            schema=NormalizedDocument,
            reasoning_effort=self.config.reasoning_effort,
        )
        problems = self._verify(f"{output.title}\n{output.body_markdown}", source)

        if problems:
            title, body = document.title, source
        else:
            title, body = output.title.strip(), output.body_markdown.strip()

        file = f"{document.id}.md"
        path = context.kb_dir / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_front_matter(document, title, normalized=not problems) + f"# {title}\n\n{body}\n")

        dropped = sorted(set(numbers(source)) - set(numbers(body)), key=lambda n: (len(n), n))
        return DocumentReport(
            id=document.id, file=file, normalized=not problems, problems=problems, dropped_numbers=dropped
        )

    def _verify(self, output: str, source: str) -> list[str]:
        """Deterministic checks that the output adds nothing the source does not say."""
        problems = []
        source_numbers = set(numbers(source))
        added_numbers = sorted({n for n in numbers(output) if n not in source_numbers})
        if added_numbers:
            problems.append(f"numbers not in source: {', '.join(added_numbers)}")

        added_emails = sorted(emails(output) - emails(source))
        if added_emails:
            problems.append(f"e-mails not in source: {', '.join(added_emails)}")

        words = stems(output)
        missing = ungrounded_words(output, source)
        coverage = 1 - len(missing) / len(words) if words else 0.0
        if coverage < self.config.min_word_coverage:
            sample = ", ".join(dict.fromkeys(missing))
            problems.append(f"word coverage {coverage:.0%}, new words: {sample[:200]}")
        return problems

    def _prompt(self, document: DocumentGroup, profile: SiteProfile, source: str) -> str:
        return "\n".join(
            [
                f"COMPANY: {profile.company_name} ({profile.business_type})",
                f"LANGUAGE: {profile.language}",
                f"DOCUMENT: {document.id} | topic: {document.topic} | working title: {document.title}",
                "",
                "SOURCE:",
                source,
            ]
        )

    def _remove_previous(self, context: SiteContext) -> None:
        """Delete files written by the last run, so renamed documents leave no stale copies."""
        report = context.normalized_dir / REPORT
        if not report.exists():
            return
        for item in json.loads(report.read_text()):
            (context.kb_dir / item["file"]).unlink(missing_ok=True)


def _front_matter(document: DocumentGroup, title: str, normalized: bool) -> str:
    # JSON values are valid YAML, which keeps quoting safe without a YAML library.
    fields = {
        "title": title,
        "topic": str(document.topic),
        "entity": document.entity_id,
        "source_urls": document.source_urls,
        "section_ids": document.section_ids,
        "normalized": normalized,
    }
    lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in fields.items()]
    return "---\n" + "\n".join(lines) + "\n---\n\n"
