from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class PageKind(StrEnum):
    PENDING = "pending"
    HTML = "html"
    PDF = "pdf"
    REDIRECT = "redirect"
    SKIPPED = "skipped"
    ERROR = "error"


class DiscoveredVia(StrEnum):
    START = "start"
    SITEMAP = "sitemap"
    LINK = "link"


class StageResult(BaseModel):
    site: str
    out_dir: Path

    def counts(self) -> dict[str, int]:
        return {}


class CrawlConfig(BaseModel):
    max_pages: int = Field(200, ge=1)
    max_depth: int = Field(5, ge=0)
    delay: float = Field(0.5, ge=0)
    timeout: float = Field(20.0, gt=0)
    refresh: bool = False


class PageRecord(BaseModel):
    url: str
    depth: int
    discovered_via: DiscoveredVia
    kind: PageKind = PageKind.PENDING
    final_url: str | None = None
    status: int | None = None
    content_type: str | None = None
    file: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    links_out: int = 0
    fetched_at: datetime | None = None
    note: str | None = None


class CrawlManifest(BaseModel):
    start_url: str
    crawled_at: datetime
    records: list[PageRecord]


class CrawlResult(StageResult):
    records: list[PageRecord]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return dict(sorted(counts.items()))


class PageFlag(StrEnum):
    EMPTY = "empty"
    THIN = "thin"
    CMS_DEFAULT = "cms_default"
    LOGIN = "login"
    DUPLICATE = "duplicate"


class Contacts(BaseModel):
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)


class ExtractConfig(BaseModel):
    thin_words: int = Field(40, ge=0)
    # A paragraph found on at least this share of HTML pages is site chrome.
    boilerplate_share: float = Field(0.6, gt=0, le=1)
    # Below this many HTML pages the share above is meaningless.
    boilerplate_min_pages: int = Field(4, ge=2)


class ExtractedPage(BaseModel):
    url: str
    kind: PageKind
    source_file: str
    file: str
    title: str | None = None
    description: str | None = None
    lang: str | None = None
    word_count: int = 0
    contacts: Contacts = Field(default_factory=Contacts)
    json_ld: list[dict] = Field(default_factory=list)
    flags: list[PageFlag] = Field(default_factory=list)
    duplicate_of: str | None = None
    boilerplate_removed: int = 0

    @property
    def is_usable(self) -> bool:
        return not self.flags


class ExtractManifest(BaseModel):
    site: str
    extracted_at: datetime
    boilerplate_file: str | None = None
    pages: list[ExtractedPage]


class ExtractResult(StageResult):
    pages: list[ExtractedPage]

    def counts(self) -> dict[str, int]:
        usable = sum(page.is_usable for page in self.pages)
        counts = {"usable": usable, "flagged": len(self.pages) - usable}
        for page in self.pages:
            for flag in page.flags:
                counts[flag] = counts.get(flag, 0) + 1
        return counts


class SectionSource(StrEnum):
    PAGE = "page"
    BOILERPLATE = "boilerplate"


class SegmentConfig(BaseModel):
    # Deeper headings are usually styled table cells or labels, not sections.
    max_heading_level: int = Field(3, ge=1, le=6)
    # Sections with less body text are merged into the next one.
    min_words: int = Field(25, ge=0)
    # Longer sections are split on paragraph boundaries.
    max_words: int = Field(500, ge=50)


class Section(BaseModel):
    id: str
    source: SectionSource
    page_url: str
    page_file: str
    page_title: str | None = None
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    order: int
    part: int = 1
    text: str
    word_count: int
    sha256: str


class SegmentManifest(BaseModel):
    site: str
    segmented_at: datetime
    pages: int
    sections: int
    skipped_pages: list[str]


class SegmentResult(StageResult):
    sections: list[Section]
    skipped_pages: list[str]

    def counts(self) -> dict[str, int]:
        return {
            "pages": len({s.page_url for s in self.sections}),
            "sections": len(self.sections),
            "skipped_pages": len(self.skipped_pages),
        }


class Audience(StrEnum):
    B2B = "b2b"
    B2C = "b2c"
    BOTH = "both"


class EntityKind(StrEnum):
    PRODUCT = "product"
    PRODUCT_LINE = "product_line"
    SERVICE = "service"


class Entity(BaseModel):
    """An offering. Built by code from site phrases the LLM selected; never free text."""

    id: str
    name: str = Field(description="Exactly as written on the site.")
    kind: EntityKind
    candidate_ids: list[str]
    section_ids: list[str]


class ChecklistItem(BaseModel):
    key: str = Field(description="snake_case English key, e.g. 'opening_hours'.")
    question: str = Field(description="A typical customer question, in the site's language.")
    why: str = Field(description="Why customers of this business ask it, one short sentence.")


class ProfileDraft(BaseModel):
    """What the LLM fills in about the company; entities are chosen separately."""

    company_name: str
    business_type: str = Field(description="What the company is, e.g. 'serwis opon', in the site's language.")
    audience: Audience
    language: str = Field(description="ISO 639-1 code of the site's main language.")
    locations: list[str] = Field(description="Cities or addresses where the company operates, copied from the input.")
    support_checklist: list[ChecklistItem]


class SiteProfile(BaseModel):
    company_name: str
    business_type: str = Field(description="What the company is, e.g. 'serwis opon', in the site's language.")
    audience: Audience
    language: str = Field(description="ISO 639-1 code of the site's main language.")
    locations: list[str] = Field(description="Cities or addresses where the company operates, copied from the input.")
    entities: list[Entity]
    support_checklist: list[ChecklistItem]


class ProfileConfig(BaseModel):
    # Words taken from the start of each section for the digest sent to the LLM.
    snippet_words: int = Field(40, ge=5)
    # Upper bound for the whole digest; snippets shrink to fit.
    max_digest_words: int = Field(4000, ge=200)
    # One call per site that shapes every later stage: worth more thinking.
    reasoning_effort: str = "medium"
    # Entity selection runs this many times; a choice needs `min_votes` of them.
    votes: int = Field(5, ge=1)
    min_votes: int = Field(3, ge=1)


class Rejection(BaseModel):
    field: str
    value: str
    reason: str


class ProfileResult(StageResult):
    profile: SiteProfile
    rejected: list[Rejection]

    def counts(self) -> dict[str, int]:
        return {
            "entities": len(self.profile.entities),
            "checklist": len(self.profile.support_checklist),
            "rejected": len(self.rejected),
        }


class Topic(StrEnum):
    OFFERING = "offering"
    PRICING = "pricing"
    CONTACT = "contact"
    COMPANY = "company"
    PROCESS = "process"
    POLICY = "policy"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    PORTFOLIO = "portfolio"
    NOISE = "noise"


class ClassifyConfig(BaseModel):
    batch_size: int = Field(8, ge=1)
    # Longer section texts are cut for the prompt; the full text is kept on disk.
    max_section_words: int = Field(300, ge=50)
    reasoning_effort: str = "low"


class ClassifiedSection(Section):
    topic: Topic | None = None
    entity_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class ClassifyResult(StageResult):
    sections: list[ClassifiedSection]
    rejected: list[Rejection]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for section in self.sections:
            key = section.topic or "unclassified"
            counts[key] = counts.get(key, 0) + 1
        counts["rejected"] = len(self.rejected)
        return counts


class MergeConfig(BaseModel):
    # An offering section naming this many entities is an overview, not about one of them.
    overview_min_entities: int = Field(3, ge=2)
    # A section whose words are this much contained in another one is a duplicate.
    duplicate_containment: float = Field(0.8, gt=0, le=1)


class DroppedSection(BaseModel):
    section_id: str
    reason: str
    duplicate_of: str | None = None


class DocumentGroup(BaseModel):
    id: str = Field(description="Future file path without extension, e.g. 'offerings/naprawa-opon'.")
    topic: Topic
    entity_id: str | None = None
    title: str
    section_ids: list[str]
    source_urls: list[str]


class MergeResult(StageResult):
    documents: list[DocumentGroup]
    dropped: list[DroppedSection]

    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "sections": sum(len(d.section_ids) for d in self.documents),
            "dropped": len(self.dropped),
        }


class NormalizeConfig(BaseModel):
    # Restructuring tables (e.g. which price list is for which vehicle) is beyond
    # gpt-5-nano; the default model stays for every other stage.
    model: str | None = "gpt-5-mini-2025-08-07"
    reasoning_effort: str = "medium"
    # Share of the output's meaningful words that must occur in the source text.
    min_word_coverage: float = Field(0.9, gt=0, le=1)


class NormalizedDocument(BaseModel):
    title: str = Field(description="Short title in the site's language, using words from the source.")
    body_markdown: str = Field(description="The document body in Markdown, without the title heading.")


class DocumentReport(BaseModel):
    id: str
    file: str
    normalized: bool
    problems: list[str] = Field(default_factory=list)
    dropped_numbers: list[str] = Field(default_factory=list)


class NormalizeResult(StageResult):
    reports: list[DocumentReport]

    def counts(self) -> dict[str, int]:
        normalized = sum(r.normalized for r in self.reports)
        return {"files": len(self.reports), "normalized": normalized, "fallback": len(self.reports) - normalized}
