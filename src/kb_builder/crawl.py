import hashlib
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.robotparser import RobotFileParser
from urllib.parse import urlsplit

import httpx
from selectolax.lexbor import LexborHTMLParser

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


@dataclass
class CrawlConfig:
    max_pages: int = 200
    max_depth: int = 5
    delay: float = 0.5
    timeout: float = 20.0
    refresh: bool = False


@dataclass
class PageRecord:
    url: str
    depth: int
    discovered_via: str  # start | sitemap | link
    kind: str = "pending"  # html | pdf | redirect | skipped | error
    final_url: str | None = None
    status: int | None = None
    content_type: str | None = None
    file: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    links_out: int = 0
    fetched_at: str | None = None
    note: str | None = None


@dataclass
class CrawlResult:
    site: str
    out_dir: Path
    records: list[PageRecord] = field(default_factory=list)


def crawl_site(start_url: str, out_root: Path, config: CrawlConfig) -> CrawlResult:
    start = normalize(start_url)
    if not start:
        raise ValueError(f"Not an http(s) URL: {start_url}")

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=config.timeout,
    ) as client:
        # Resolve redirects first (e.g. bare domain -> www) so robots.txt and
        # the sitemap are read from the host that actually serves the site.
        origin = _origin(str(client.get(start).url))
        site = site_key(urlsplit(origin).netloc)
        out_dir = out_root / site
        (out_dir / "pages").mkdir(parents=True, exist_ok=True)

        robots = _load_robots(client, origin)
        cached = {} if config.refresh else _load_manifest(out_dir)

        queue: deque[PageRecord] = deque([PageRecord(start, 0, "start")])
        for url in _sitemap_urls(client, origin, robots):
            queue.append(PageRecord(url, 1, "sitemap"))

        seen: set[str] = set()
        records: list[PageRecord] = []
        fetched = 0

        while queue:
            record = queue.popleft()
            if record.url in seen:
                continue
            seen.add(record.url)

            skip = _skip_reason(record.url, site, robots)
            if skip:
                record.kind, record.note = "skipped", skip
                records.append(record)
                continue
            if fetched >= config.max_pages:
                record.kind, record.note = "skipped", "max_pages reached"
                records.append(record)
                continue

            body, from_cache = _fetch(client, record, out_dir, cached.get(record.url))
            records.append(record)
            if not from_cache:
                fetched += 1
                time.sleep(config.delay)
            _log(record, from_cache)

            if record.kind == "redirect":
                if record.final_url not in seen:
                    queue.appendleft(PageRecord(record.final_url, record.depth, record.discovered_via))
                continue
            if record.kind != "html" or body is None or record.depth >= config.max_depth:
                continue

            links = _links(body, record.final_url or record.url)
            record.links_out = len(links)
            for link in links:
                if link not in seen:
                    queue.append(PageRecord(link, record.depth + 1, "link"))

    _save_manifest(out_dir, start, records)
    return CrawlResult(site=site, out_dir=out_dir, records=records)


def _fetch(
    client: httpx.Client,
    record: PageRecord,
    out_dir: Path,
    cached: dict | None,
) -> tuple[bytes | None, bool]:
    """Fill `record` from the cache or the network; return (body, from_cache)."""
    if cached and cached.get("file") and (out_dir / cached["file"]).exists():
        for key, value in cached.items():
            if key not in ("depth", "discovered_via"):
                setattr(record, key, value)
        return (out_dir / record.file).read_bytes(), True

    record.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        response = client.get(record.url)
    except httpx.HTTPError as exc:
        record.kind, record.note = "error", f"{type(exc).__name__}: {exc}"
        return None, False

    record.status = response.status_code
    record.content_type = response.headers.get("content-type", "").split(";")[0].strip()
    final = normalize(str(response.url))
    record.final_url = final

    if final and final != record.url:
        # Store the page once, under its final URL.
        record.kind, record.note = "redirect", f"-> {final}"
        return None, False
    if response.status_code >= 400:
        record.kind, record.note = "error", f"HTTP {response.status_code}"
        return None, False

    if "html" in record.content_type:
        record.kind, suffix = "html", ".html"
    elif "pdf" in record.content_type or extension(record.url) == ".pdf":
        record.kind, suffix = "pdf", ".pdf"
    else:
        record.kind, record.note = "skipped", f"content-type {record.content_type}"
        return None, False

    body = response.content
    record.file = f"pages/{cache_name(record.url)}{suffix}"
    record.sha256 = hashlib.sha256(body).hexdigest()
    record.bytes = len(body)
    (out_dir / record.file).write_bytes(body)
    return body, False


def _skip_reason(url: str, site: str, robots: RobotFileParser) -> str | None:
    if not is_same_site(url, site):
        return "external"
    ext = extension(url)
    if ext in SKIP_EXTENSIONS:
        return f"asset {ext}"
    if ext and ext not in DOCUMENT_EXTENSIONS and ext not in (".html", ".htm", ".php"):
        return f"unsupported {ext}"
    if not robots.can_fetch(USER_AGENT, url):
        return "robots.txt"
    return None


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


def _load_robots(client: httpx.Client, origin: str) -> RobotFileParser:
    robots = RobotFileParser()
    try:
        response = client.get(f"{origin}/robots.txt")
        lines = response.text.splitlines() if response.status_code == 200 else []
    except httpx.HTTPError:
        lines = []
    robots.parse(lines)
    return robots


def _sitemap_urls(client: httpx.Client, origin: str, robots: RobotFileParser) -> list[str]:
    """Page URLs from all sitemaps, following sitemap indexes."""
    pending = list(robots.site_maps() or []) or [f"{origin}/sitemap.xml"]
    visited: set[str] = set()
    pages: list[str] = []

    while pending and len(visited) < MAX_SITEMAPS:
        sitemap = pending.pop(0)
        if sitemap in visited:
            continue
        visited.add(sitemap)
        try:
            response = client.get(sitemap)
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


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _load_manifest(out_dir: Path) -> dict[str, dict]:
    path = out_dir / "crawl.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {r["url"]: r for r in data["records"] if r.get("file")}


def _save_manifest(out_dir: Path, start: str, records: list[PageRecord]) -> None:
    data = {
        "start_url": start,
        "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "records": [asdict(r) for r in records],
    }
    (out_dir / "crawl.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _log(record: PageRecord, from_cache: bool) -> None:
    source = "cache" if from_cache else (str(record.status) if record.status else "-")
    note = f"  ({record.note})" if record.note else ""
    print(f"[{record.kind:8}] {source:>5}  d={record.depth}  {record.url}{note}")
