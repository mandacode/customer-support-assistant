import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from markdownify import markdownify
from pypdf import PdfReader
from selectolax.lexbor import LexborHTMLParser, LexborNode

from kb_builder.models import (
    Contacts,
    CrawlManifest,
    ExtractConfig,
    ExtractedPage,
    ExtractManifest,
    ExtractResult,
    PageFlag,
    PageKind,
    PageRecord,
)
from kb_builder.services.base import BaseService

MANIFEST = "extract.json"
BOILERPLATE = "boilerplate.md"

# Never content: removed before conversion.
NOISE_SELECTORS = [
    "script", "style", "noscript", "template", "svg", "iframe", "img", "picture", "video",
    "form", "select", "button", "input", "textarea",
    ".skip-link", ".screen-reader-text", ".sr-only", ".visually-hidden",
]

# Site chrome: removed from pages, but kept once in boilerplate.md because it
# often holds the only copy of contact details.
CHROME_SELECTORS = [
    "nav", "header", "footer", "aside",
    "[role=navigation]", "[role=banner]", "[role=contentinfo]", "[role=dialog]",
    "[data-elementor-type=header]", "[data-elementor-type=footer]", "[data-elementor-type=popup]",
]

# Default content shipped with CMS installs.
CMS_DEFAULT_PHRASES = [
    "welcome to wordpress",
    "this is your first post",
    "this is an example page",
    "witaj w wordpressie",
    "to jest przykładowa strona",
    "just another wordpress site",
]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# Shorter blocks are table cells, labels or prices: repeating them is meaningful.
DEDUPE_MIN_WORDS = 8


class ExtractService(BaseService[ExtractResult]):
    """Turn cached raw pages into clean Markdown plus per-page metadata."""

    def __init__(self, raw_dir: Path, out_root: Path, config: ExtractConfig | None = None) -> None:
        super().__init__()
        self.raw_dir = raw_dir
        self.site = raw_dir.name
        self.out_dir = out_root / self.site
        self.config = config or ExtractConfig()

    def run(self) -> ExtractResult:
        crawl = CrawlManifest.model_validate_json((self.raw_dir / "crawl.json").read_text())
        records = [r for r in crawl.records if r.kind in (PageKind.HTML, PageKind.PDF) and r.file]

        pages: list[ExtractedPage] = []
        bodies: list[list[str]] = []
        chrome: list[str] = []
        for record in records:
            page, paragraphs, page_chrome = self._extract(record)
            pages.append(page)
            bodies.append(paragraphs)
            if record.url == crawl.start_url:
                chrome = page_chrome

        shared = self._shared_paragraphs(pages, bodies)
        shared_in_order = [p for body in bodies for p in body if p in shared]
        bodies = [self._drop_shared(page, body, shared) for page, body in zip(pages, bodies)]

        (self.out_dir / "pages").mkdir(parents=True, exist_ok=True)
        seen_hashes: dict[str, str] = {}
        for page, body in zip(pages, bodies):
            markdown = "\n\n".join(body)
            page.word_count = len(re.findall(r"\w+", markdown))
            self._flag(page, markdown, seen_hashes)
            (self.out_dir / page.file).write_text(markdown + "\n")
            self._log(page)

        boilerplate = _unique(chrome + shared_in_order)
        boilerplate_file = None
        if boilerplate:
            boilerplate_file = BOILERPLATE
            (self.out_dir / BOILERPLATE).write_text("\n\n".join(boilerplate) + "\n")

        manifest = ExtractManifest(
            site=self.site,
            extracted_at=datetime.now(timezone.utc).replace(microsecond=0),
            boilerplate_file=boilerplate_file,
            pages=pages,
        )
        (self.out_dir / MANIFEST).write_text(manifest.model_dump_json(indent=2))
        return ExtractResult(site=self.site, out_dir=self.out_dir, pages=pages)

    def _extract(self, record: PageRecord) -> tuple[ExtractedPage, list[str], list[str]]:
        """Return (page metadata, body paragraphs, chrome paragraphs)."""
        raw = (self.raw_dir / record.file).read_bytes()
        page = ExtractedPage(
            url=record.final_url or record.url,
            kind=record.kind,
            source_file=record.file,
            file=f"pages/{Path(record.file).stem}.md",
        )
        if record.kind == PageKind.PDF:
            return page, _pdf_paragraphs(raw), []

        tree = LexborHTMLParser(raw.decode("utf-8", errors="replace"))
        page.title = _text(tree.css_first("title"))
        page.description = _meta(tree, "description") or _meta(tree, "og:description")
        page.lang = (tree.css_first("html").attributes.get("lang") if tree.css_first("html") else None)
        page.json_ld = _json_ld(tree)
        page.contacts = _contacts(tree)
        if tree.css_first("input[type=password]"):
            page.flags.append(PageFlag.LOGIN)

        _remove(tree, NOISE_SELECTORS)
        chrome = []
        for selector in CHROME_SELECTORS:
            for node in tree.css(selector):
                chrome.extend(_paragraphs(node.html or ""))
                node.decompose()

        root = tree.css_first("main") or tree.body
        body = _paragraphs(root.html or "") if root else []
        return page, _unique(body), _unique(chrome)

    def _shared_paragraphs(self, pages: list[ExtractedPage], bodies: list[list[str]]) -> set[str]:
        """Paragraphs repeated on most HTML pages (CTA blocks, contact strips...)."""
        html_bodies = [set(b) for p, b in zip(pages, bodies) if p.kind == PageKind.HTML]
        if len(html_bodies) < self.config.boilerplate_min_pages:
            return set()
        counts = Counter(paragraph for body in html_bodies for paragraph in body)
        threshold = self.config.boilerplate_share * len(html_bodies)
        return {paragraph for paragraph, n in counts.items() if n >= threshold}

    def _drop_shared(self, page: ExtractedPage, body: list[str], shared: set[str]) -> list[str]:
        kept = [p for p in body if p not in shared]
        page.boilerplate_removed = len(body) - len(kept)
        return kept

    def _flag(self, page: ExtractedPage, markdown: str, seen_hashes: dict[str, str]) -> None:
        if page.word_count == 0:
            page.flags.append(PageFlag.EMPTY)
        elif page.word_count < self.config.thin_words:
            page.flags.append(PageFlag.THIN)

        lowered = markdown.lower()
        if any(phrase in lowered for phrase in CMS_DEFAULT_PHRASES):
            page.flags.append(PageFlag.CMS_DEFAULT)

        digest = hashlib.sha256(markdown.encode()).hexdigest()
        if markdown and digest in seen_hashes:
            page.flags.append(PageFlag.DUPLICATE)
            page.duplicate_of = seen_hashes[digest]
        else:
            seen_hashes.setdefault(digest, page.url)

    def _log(self, page: ExtractedPage) -> None:
        flags = ",".join(page.flags) or "ok"
        self.logger.info(
            "[%-4s] %5d words  -%d shared  %-22s %s",
            page.kind, page.word_count, page.boilerplate_removed, flags, page.url,
        )


def _paragraphs(html: str) -> list[str]:
    markdown = markdownify(html, heading_style="ATX", bullets="-", strip=["a"])
    paragraphs = []
    for block in re.split(r"\n\s*\n", markdown):
        lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in block.splitlines()]
        text = "\n".join(line for line in lines if line)
        if re.search(r"\w", text):
            paragraphs.append(text)
    return paragraphs


def _pdf_paragraphs(raw: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(raw))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for page in reader.pages for line in (page.extract_text() or "").splitlines()]
    # PDF text comes hard-wrapped; a line not ending a sentence continues the paragraph.
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if not line:
            continue
        current = f"{current} {line}".strip()
        if re.search(r"[.:;!?]$", line):
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return _unique(paragraphs)


def _unique(paragraphs: list[str]) -> list[str]:
    """Drop repeated long blocks, e.g. desktop and mobile copies of the same text."""
    seen: set[str] = set()
    result = []
    for paragraph in paragraphs:
        key = re.sub(r"\W+", " ", paragraph).strip().lower()
        if not key:
            continue
        if len(key.split()) < DEDUPE_MIN_WORDS:
            result.append(paragraph)
        elif key not in seen:
            seen.add(key)
            result.append(paragraph)
    return result


def _remove(tree: LexborHTMLParser, selectors: list[str]) -> None:
    for selector in selectors:
        for node in tree.css(selector):
            node.decompose()


def _text(node: LexborNode | None) -> str | None:
    return node.text(strip=True) or None if node else None


def _meta(tree: LexborHTMLParser, name: str) -> str | None:
    node = tree.css_first(f'meta[name="{name}"]') or tree.css_first(f'meta[property="{name}"]')
    return (node.attributes.get("content") or None) if node else None


def _json_ld(tree: LexborHTMLParser) -> list[dict]:
    items = []
    for node in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.text())
        except json.JSONDecodeError:
            continue
        items.extend(data if isinstance(data, list) else [data])
    return [item for item in items if isinstance(item, dict)]


def _contacts(tree: LexborHTMLParser) -> Contacts:
    emails: list[str] = []
    phones: list[str] = []
    for node in tree.css("a[href]"):
        href = unquote(node.attributes.get("href") or "").strip()
        if href.lower().startswith("mailto:"):
            emails.append(href[7:].split("?")[0].strip().lower())
        elif href.lower().startswith("tel:"):
            phones.append(re.sub(r"[^\d+]", "", href[4:]))
    body = tree.body.text(separator=" ") if tree.body else ""
    emails.extend(email.lower() for email in EMAIL_RE.findall(body))
    return Contacts(emails=sorted(set(e for e in emails if e)), phones=sorted(set(p for p in phones if p)))
