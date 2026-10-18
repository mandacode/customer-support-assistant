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


class CrawlResult(BaseModel):
    site: str
    out_dir: Path
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


class ExtractResult(BaseModel):
    site: str
    out_dir: Path
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


class SegmentResult(BaseModel):
    site: str
    out_dir: Path
    sections: list[Section]
    skipped_pages: list[str]

    def counts(self) -> dict[str, int]:
        return {
            "pages": len({s.page_url for s in self.sections}),
            "sections": len(self.sections),
            "skipped_pages": len(self.skipped_pages),
        }
