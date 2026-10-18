import re
from enum import StrEnum

from pydantic import BaseModel, Field

from kb_builder.grounding import fold
from kb_builder.models import ExtractManifest, Section, SectionSource

# Offerings are named in short phrases; longer blocks are sentences.
MAX_WORDS = 6
# JSON-LD types whose `name` is an offering.
OFFERING_TYPES = {"Product", "Service", "Offer", "ProductModel", "IndividualProduct"}

HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+(.+)$")
PRICE_RE = re.compile(r"\d\s*(?:pln|zł|zl|eur|€|\$|usd|gbp|£)", re.I)
ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")


class CandidateKind(StrEnum):
    HEADING = "heading"
    LABEL = "label"  # bold text or a short standalone line
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    JSON_LD = "json_ld"


class Candidate(BaseModel):
    id: str
    text: str = Field(description="Exactly as written on the site (first occurrence).")
    kinds: list[CandidateKind]
    section_ids: list[str]


def extract_candidates(sections: list[Section], extract: ExtractManifest) -> list[Candidate]:
    """Short phrases that may name an offering, found deterministically in the site text.

    Sources: headings, bold labels, short standalone lines (e.g. table cells a
    page builder renders as blocks), list items, the first cell of Markdown
    table rows, and Product/Service names in JSON-LD. Site chrome is skipped.
    Phrases are merged by their folded form, keeping where each one occurs.
    """
    found: dict[str, Candidate] = {}

    def add(text: str, kind: CandidateKind, section_id: str | None) -> None:
        text = re.sub(r"\s+", " ", text.replace("**", "")).strip(" :–-")
        if not _looks_like_name(text):
            return
        key = re.sub(r"\W+", " ", fold(text)).strip()
        candidate = found.get(key)
        if candidate is None:
            candidate = found[key] = Candidate(id="", text=text, kinds=[], section_ids=[])
        if kind not in candidate.kinds:
            candidate.kinds.append(kind)
        if section_id and section_id not in candidate.section_ids:
            candidate.section_ids.append(section_id)

    for section in sections:
        if section.source == SectionSource.BOILERPLATE:
            continue
        for match in BOLD_RE.finditer(section.text):
            add(match.group(1), CandidateKind.LABEL, section.id)
        for block in re.split(r"\n\s*\n", section.text):
            lines = [line for line in block.strip().splitlines() if line.strip()]
            for line in lines:
                if heading := HEADING_RE.match(line):
                    add(heading.group(1), CandidateKind.HEADING, section.id)
                elif item := LIST_RE.match(line):
                    add(item.group(1), CandidateKind.LIST_ITEM, section.id)
                elif row := ROW_RE.match(line):
                    first_cell = row.group(1).split("|")[0]
                    if not re.fullmatch(r"\s*:?-{3,}:?\s*", first_cell):
                        add(first_cell, CandidateKind.TABLE_CELL, section.id)
            if len(lines) == 1 and not HEADING_RE.match(lines[0]) and not LIST_RE.match(lines[0]):
                add(lines[0], CandidateKind.LABEL, section.id)

    for page in extract.pages:
        if not page.is_usable:
            continue
        for item in _json_ld_items(page.json_ld):
            types = item.get("@type", [])
            if OFFERING_TYPES & set(types if isinstance(types, list) else [types]) and item.get("name"):
                add(str(item["name"]), CandidateKind.JSON_LD, None)

    candidates = list(found.values())
    for index, candidate in enumerate(candidates, start=1):
        candidate.id = f"c{index:02d}"
    return candidates


def _looks_like_name(text: str) -> bool:
    words = re.findall(r"\w+", text)
    letters = [w for w in words if not w.isdigit()]
    if not letters or len(words) > MAX_WORDS:
        return False
    if not any(len(w) >= 3 for w in letters):
        return False
    # Prices, dimensions and phone numbers: mostly digits.
    if len(words) - len(letters) >= len(letters):
        return False
    # Prices, e-mail addresses and links are values, not names.
    if PRICE_RE.search(text) or "@" in text or "://" in text:
        return False
    # Questions and exclamations are slogans ("Twoja opona jest uszkodzona?"),
    # trailing punctuation marks a fragment of a sentence or an enumeration.
    return not text.endswith(("?", "!", ".", ";", ","))


def _json_ld_items(blocks: list[dict]) -> list[dict]:
    items: list[dict] = []
    for block in blocks:
        for item in block.get("@graph", [block]):
            if isinstance(item, dict):
                items.append(item)
                offers = item.get("offers") or item.get("hasOfferCatalog", {}).get("itemListElement", [])
                items.extend(o for o in (offers if isinstance(offers, list) else [offers]) if isinstance(o, dict))
    return items
