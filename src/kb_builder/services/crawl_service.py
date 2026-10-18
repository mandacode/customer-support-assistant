import hashlib
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from selectolax.lexbor import LexborHTMLParser

from kb_builder.models import CrawlConfig, CrawlManifest, CrawlResult, DiscoveredVia, PageKind, PageRecord
from kb_builder.services.base_service import BaseService
from kb_builder.urls import (
    DOCUMENT_EXTENSIONS,
    SKIP_EXTENSIONS,
    cache_name,
    extension,
    is_same_site,
    normalize,
    site_key,
)

USER_AGENT = "kb-builder/0.1 (+customer-support-assistant)"
MAX_SITEMAPS = 20
MANIFEST = "crawl.json"


class CrawlService(BaseService[CrawlResult]):
    """Discover a site's pages (sitemap + internal links) and cache raw HTML/PDF."""

    def __init__(self, start_url: str, out_root: Path, config: CrawlConfig | None = None) -> None:
        super().__init__()
        start = normalize(start_url)
        if not start:
            raise ValueError(f"Not an http(s) URL: {start_url}")
        self.start_url = start
        self.out_root = out_root
        self.config = config or CrawlConfig()

    def run(self) -> CrawlResult:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=self.config.timeout,
        ) as client:
            self._client = client
            # Resolve redirects first (e.g. bare domain -> www) so robots.txt and
            # the sitemap are read from the host that actually serves the site.
            origin = _origin(str(client.get(self.start_url).url))
            site = site_key(urlsplit(origin).netloc)
            self._site = site
            self._out_dir = self.out_root / site
            (self._out_dir / "pages").mkdir(parents=True, exist_ok=True)

            self._robots = self._load_robots(origin)
            self._cached = {} if self.config.refresh else self._load_manifest()
            records = self._crawl(origin)

        self._save_manifest(records)
        return CrawlResult(site=site, out_dir=self._out_dir, records=records)

    def _crawl(self, origin: str) -> list[PageRecord]:
        queue: deque[PageRecord] = deque([PageRecord(url=self.start_url, depth=0, discovered_via=DiscoveredVia.START)])
        for url in self._sitemap_urls(origin):
            queue.append(PageRecord(url=url, depth=1, discovered_via=DiscoveredVia.SITEMAP))

        seen: set[str] = set()
        records: list[PageRecord] = []
        fetched = 0

        while queue:
            record = queue.popleft()
            if record.url in seen:
                continue
            seen.add(record.url)

            skip = self._skip_reason(record.url)
            if not skip and fetched >= self.config.max_pages:
                skip = "max_pages reached"
            if skip:
                record.kind, record.note = PageKind.SKIPPED, skip
                records.append(record)
                continue

            record, body, from_cache = self._fetch(record)
            records.append(record)
            if not from_cache:
                fetched += 1
                time.sleep(self.config.delay)
            self._log(record, from_cache)

            if record.kind == PageKind.REDIRECT:
                if record.final_url not in seen:
                    queue.appendleft(
                        PageRecord(url=record.final_url, depth=record.depth, discovered_via=record.discovered_via)
                    )
                continue
            if record.kind != PageKind.HTML or body is None or record.depth >= self.config.max_depth:
                continue

            links = _links(body, record.final_url or record.url)
            record.links_out = len(links)
            for link in links:
                if link not in seen:
                    queue.append(PageRecord(url=link, depth=record.depth + 1, discovered_via=DiscoveredVia.LINK))

        return records

    def _fetch(self, record: PageRecord) -> tuple[PageRecord, bytes | None, bool]:
        """Fill `record` from the cache or the network; return (record, body, from_cache)."""
        cached = self._cached.get(record.url)
        if cached and cached.file and (self._out_dir / cached.file).exists():
            record = cached.model_copy(update={"depth": record.depth, "discovered_via": record.discovered_via})
            return record, (self._out_dir / record.file).read_bytes(), True

        record.fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
        try:
            response = self._client.get(record.url)
        except httpx.HTTPError as exc:
            record.kind, record.note = PageKind.ERROR, f"{type(exc).__name__}: {exc}"
            return record, None, False

        record.status = response.status_code
        record.content_type = response.headers.get("content-type", "").split(";")[0].strip()
        final = normalize(str(response.url))
        record.final_url = final

        if final and final != record.url:
            # Store the page once, under its final URL.
            record.kind, record.note = PageKind.REDIRECT, f"-> {final}"
            return record, None, False
        if response.status_code >= 400:
            record.kind, record.note = PageKind.ERROR, f"HTTP {response.status_code}"
            return record, None, False

        if "html" in record.content_type:
            record.kind, suffix = PageKind.HTML, ".html"
        elif "pdf" in record.content_type or extension(record.url) == ".pdf":
            record.kind, suffix = PageKind.PDF, ".pdf"
        else:
            record.kind, record.note = PageKind.SKIPPED, f"content-type {record.content_type}"
            return record, None, False

        body = response.content
        record.file = f"pages/{cache_name(record.url)}{suffix}"
        record.sha256 = hashlib.sha256(body).hexdigest()
        record.bytes = len(body)
        (self._out_dir / record.file).write_bytes(body)
        return record, body, False

    def _skip_reason(self, url: str) -> str | None:
        if not is_same_site(url, self._site):
            return "external"
        ext = extension(url)
        if ext in SKIP_EXTENSIONS:
            return f"asset {ext}"
        if ext and ext not in DOCUMENT_EXTENSIONS and ext not in (".html", ".htm", ".php"):
            return f"unsupported {ext}"
        if not self._robots.can_fetch(USER_AGENT, url):
            return "robots.txt"
        return None

    def _load_robots(self, origin: str) -> RobotFileParser:
        robots = RobotFileParser()
        try:
            response = self._client.get(f"{origin}/robots.txt")
            lines = response.text.splitlines() if response.status_code == 200 else []
        except httpx.HTTPError:
            lines = []
        robots.parse(lines)
        return robots

    def _sitemap_urls(self, origin: str) -> list[str]:
        """Page URLs from all sitemaps, following sitemap indexes."""
        pending = list(self._robots.site_maps() or []) or [f"{origin}/sitemap.xml"]
        visited: set[str] = set()
        pages: list[str] = []

        while pending and len(visited) < MAX_SITEMAPS:
            sitemap = pending.pop(0)
            if sitemap in visited:
                continue
            visited.add(sitemap)
            try:
                response = self._client.get(sitemap)
            except httpx.HTTPError:
                continue
            if response.status_code != 200:
                continue

            locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)
            if "<sitemapindex" in response.text:
                pending.extend(locs)
                continue
            for loc in locs:
                url = normalize(loc)
                if url and url not in pages:
                    pages.append(url)
        return pages

    def _load_manifest(self) -> dict[str, PageRecord]:
        path = self._out_dir / MANIFEST
        if not path.exists():
            return {}
        manifest = CrawlManifest.model_validate_json(path.read_text())
        return {r.url: r for r in manifest.records if r.file}

    def _save_manifest(self, records: list[PageRecord]) -> None:
        manifest = CrawlManifest(
            start_url=self.start_url,
            crawled_at=datetime.now(timezone.utc).replace(microsecond=0),
            records=records,
        )
        (self._out_dir / MANIFEST).write_text(manifest.model_dump_json(indent=2))

    def _log(self, record: PageRecord, from_cache: bool) -> None:
        source = "cache" if from_cache else (str(record.status) if record.status else "-")
        note = f"  ({record.note})" if record.note else ""
        self.logger.info("[%-8s] %5s  d=%d  %s%s", record.kind, source, record.depth, record.url, note)


def _links(body: bytes, base: str) -> list[str]:
    tree = LexborHTMLParser(body.decode("utf-8", errors="replace"))
    links: list[str] = []
    for node in tree.css("a[href]"):
        href = node.attributes.get("href") or ""
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        url = normalize(href, base)
        if url and url not in links:
            links.append(url)
    return links


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"
