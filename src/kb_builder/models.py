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
