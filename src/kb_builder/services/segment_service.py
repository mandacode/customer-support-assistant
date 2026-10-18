import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from kb_builder.models import (
    ExtractManifest,
    Section,
    SectionSource,
    SegmentConfig,
    SegmentManifest,
    SegmentResult,
)
from kb_builder.services.base_service import BaseService

MANIFEST = "segment.json"
SECTIONS = "sections.jsonl"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")

# A section with less body text than this is just a title for what follows.
HEADING_ONLY_WORDS = 3


@dataclass
class _Draft:
    heading: str | None
    heading_path: list[str]
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


class SegmentService(BaseService[SegmentResult]):
    """Split extracted pages into heading-based sections."""

    def __init__(self, extracted_dir: Path, out_root: Path, config: SegmentConfig | None = None) -> None:
        super().__init__()
        self.extracted_dir = extracted_dir
        self.site = extracted_dir.name
        self.out_dir = out_root / self.site
        self.config = config or SegmentConfig()

    def run(self) -> SegmentResult:
        extract = ExtractManifest.model_validate_json((self.extracted_dir / "extract.json").read_text())

        sections: list[Section] = []
        skipped: list[str] = []
        for page in extract.pages:
            if not page.is_usable:
                skipped.append(page.url)
                self.logger.info("[skip]  %-22s %s", ",".join(page.flags), page.url)
                continue
            markdown = (self.extracted_dir / page.file).read_text()
            page_sections = self._segment(
                markdown, SectionSource.PAGE, page.url, page.file, page.title, Path(page.file).stem
            )
            sections.extend(page_sections)
            self.logger.info("[page]  %3d sections  %s", len(page_sections), page.url)

        if extract.boilerplate_file:
            markdown = (self.extracted_dir / extract.boilerplate_file).read_text()
            # Site chrome has no page of its own; attribute it to the start page.
            chrome_url = f"{extract.pages[0].url}#boilerplate"
            chrome = self._segment(
                markdown, SectionSource.BOILERPLATE, chrome_url, extract.boilerplate_file, None, "boilerplate"
            )
            sections.extend(chrome)
            self.logger.info("[chrome] %3d sections  %s", len(chrome), extract.boilerplate_file)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / SECTIONS).open("w") as file:
            for section in sections:
                file.write(section.model_dump_json() + "\n")

        manifest = SegmentManifest(
            site=self.site,
            segmented_at=datetime.now(timezone.utc).replace(microsecond=0),
            pages=len({s.page_url for s in sections}),
            sections=len(sections),
            skipped_pages=skipped,
        )
        (self.out_dir / MANIFEST).write_text(manifest.model_dump_json(indent=2))
        return SegmentResult(site=self.site, out_dir=self.out_dir, sections=sections, skipped_pages=skipped)

    def _segment(
        self,
        markdown: str,
        source: SectionSource,
        page_url: str,
        page_file: str,
        page_title: str | None,
        page_key: str,
    ) -> list[Section]:
        drafts = self._merge_short(self._split_on_headings(markdown))

        sections: list[Section] = []
        for order, draft in enumerate(drafts, start=1):
            parts = self._split_long(draft.text)
            for part, text in enumerate(parts, start=1):
                slug = _slug(draft.heading) or "intro"
                suffix = f"-p{part}" if len(parts) > 1 else ""
                sections.append(
                    Section(
                        id=f"{page_key}#{order:02d}-{slug}{suffix}",
                        source=source,
                        page_url=page_url,
                        page_file=page_file,
                        page_title=page_title,
                        heading=draft.heading,
                        heading_path=draft.heading_path,
                        order=order,
                        part=part,
                        text=text,
                        word_count=_words(text),
                        sha256=hashlib.sha256(text.encode()).hexdigest(),
                    )
                )
        return sections

    def _split_on_headings(self, markdown: str) -> list[_Draft]:
        drafts = [_Draft(heading=None, heading_path=[])]
        stack: list[tuple[int, str]] = []
        lines = markdown.splitlines()
        # Page builders use several h1s as big section titles; treating each as
        # the page root would nest the rest of the page under it.
        h1_count = sum(1 for line in lines if (m := HEADING_RE.match(line)) and len(m.group(1)) == 1)

        for line in lines:
            match = HEADING_RE.match(line)
            if not match:
                drafts[-1].lines.append(line)
                continue
            level, title = len(match.group(1)), match.group(2).strip()
            if level == 1 and h1_count > 1:
                level = 2
            if level > self.config.max_heading_level:
                # Keep it as plain text: a cell or label inside the current section.
                drafts[-1].lines.append(title)
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            drafts.append(_Draft(heading=title, heading_path=[t for _, t in stack], lines=[line]))

        return [d for d in drafts if d.text]

    def _merge_short(self, drafts: list[_Draft]) -> list[_Draft]:
        """Carry heading-only sections, and short ones into their next sibling.

        A short section is only merged forward when the next section shares its
        parent heading (e.g. "Adres", "Telefon", "Email" under "Kontakt"), so a
        small but self-contained topic never ends up inside an unrelated one.
        """
        merged: list[_Draft] = []
        carry: list[str] = []
        carry_path: list[str] = []
        for index, draft in enumerate(drafts):
            own_path = draft.heading_path
            if carry:
                # Name the merged section after the deepest heading both share.
                common = _common_prefix(carry_path, draft.heading_path)
                if common:
                    draft.heading_path, draft.heading = common, common[-1]
                draft.lines = carry + [""] + draft.lines
                carry = []
            body_words = _words("\n".join(line for line in draft.lines if not HEADING_RE.match(line)))
            following = drafts[index + 1] if index + 1 < len(drafts) else None
            heading_only = body_words < HEADING_ONLY_WORDS
            short_sibling = (
                body_words < self.config.min_words
                and following is not None
                and _is_prefix(own_path[:-1], following.heading_path)
            )
            if following is not None and (heading_only or short_sibling):
                carry, carry_path = draft.lines, draft.heading_path
                continue
            merged.append(draft)
        return merged

    def _split_long(self, text: str) -> list[str]:
        if _words(text) <= self.config.max_words:
            return [text]
        parts: list[str] = []
        current: list[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            if current and _words("\n\n".join(current + [paragraph])) > self.config.max_words:
                parts.append("\n\n".join(current))
                current = []
            current.append(paragraph)
        if current:
            parts.append("\n\n".join(current))
        return parts


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _is_prefix(prefix: list[str], path: list[str]) -> bool:
    return path[: len(prefix)] == prefix


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    common = []
    for x, y in zip(a, b):
        if x != y:
            break
        common.append(x)
    return common


def _slug(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"[^\w]+", "-", text.lower()).strip("-")[:40]
