import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kb_builder.candidates import Candidate, extract_candidates
from kb_builder.context import SiteContext
from kb_builder.entity_votes import SELECTION_PROMPT, aggregate, render_candidates, selection_schema
from kb_builder.grounding import is_grounded
from kb_builder.llm import LLMClient
from kb_builder.models import (
    Entity,
    ExtractManifest,
    ProfileConfig,
    ProfileDraft,
    ProfileResult,
    Rejection,
    Section,
    SiteProfile,
)
from kb_builder.services.base_service import BaseService
from kb_builder.services.segment_service import SECTIONS, read_sections

PROFILE = "profile.json"
REJECTED = "rejected.json"
CANDIDATES = "candidates.json"
VOTES = "votes.json"
DIGEST = "digest.txt"

# Share of the business type's words that must occur in the site text.
BUSINESS_TYPE_RATIO = 0.5

# JSON-LD types that describe the business itself.
BUSINESS_TYPES = {"Organization", "LocalBusiness", "Corporation", "Store", "AutoRepair", "ProfessionalService"}
JSON_LD_DROP = {"@id", "@context", "image", "logo", "potentialAction", "isPartOf", "primaryImageOfPage"}

SYSTEM_PROMPT = """\
You analyse a company website to prepare a customer-support knowledge base.
The input is a digest: every section of the site as `[section_id] heading path :: opening words`,
plus structured data and contact links found in the HTML.

Rules:
- Use only facts present in the input. Never invent names, prices, places or services.
- Write every human-readable field in the site's main language.
- company_name: the brand name only, without slogans or city taglines from the page title.

support_checklist - 12-15 questions customers of THIS type of business typically ask
support, based on how such businesses work, NOT on what this site happens to say:
- Cover practical topics: opening hours, prices, waiting/lead time, booking, payment
  methods, warranty and complaints, delivery or service area, plus questions specific
  to the industry (e.g. tyre storage for a tyre service, certificates for a manufacturer).
- Most questions should be ones the site may NOT answer; they are used to find gaps.
- key: snake_case English, e.g. opening_hours.
"""


class ProfileService(BaseService[ProfileResult]):
    """Describe the company, its offerings and what its customers ask.

    Company fields and the support checklist come from one LLM call, checked
    against the site text. Offerings are never written by the model: code lists
    candidate phrases from the site, the model only selects and groups them by
    id (several independent votes), and code builds the entities from the
    candidates the votes agree on.
    """

    name = "profile"

    def __init__(self, config: ProfileConfig | None = None, llm: LLMClient | None = None) -> None:
        super().__init__()
        self.config = config or ProfileConfig()
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def inputs(self, context: SiteContext) -> list[Path]:
        return [context.segmented_dir / SECTIONS, context.extracted_dir / "extract.json"]

    def run(self, context: SiteContext) -> ProfileResult:
        sections = read_sections(context.segmented_dir / SECTIONS)
        extract = ExtractManifest.model_validate_json((context.extracted_dir / "extract.json").read_text())

        digest = self._digest(context.site, sections, extract)
        out_dir = context.profile_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / DIGEST).write_text(digest)
        self.logger.info("digest: %d sections, %d words", len(sections), _words(digest))

        draft = self.llm.parse(
            system=SYSTEM_PROMPT, user=digest, schema=ProfileDraft, reasoning_effort=self.config.reasoning_effort
        )
        draft, rejected = self._ground(draft, digest)

        candidates = extract_candidates(sections, extract)
        (out_dir / CANDIDATES).write_text(
            json.dumps([c.model_dump(mode="json") for c in candidates], indent=2, ensure_ascii=False)
        )
        entities = self._select_entities(digest, candidates, out_dir)
        profile = SiteProfile(**draft.model_dump(), entities=entities)

        (out_dir / PROFILE).write_text(profile.model_dump_json(indent=2))
        (out_dir / REJECTED).write_text(
            json.dumps([r.model_dump() for r in rejected], indent=2, ensure_ascii=False)
        )
        for entity in profile.entities:
            self.logger.info("  [%-12s] %-35s %d sections", entity.kind, entity.name, len(entity.section_ids))
        for rejection in rejected:
            self.logger.warning("  rejected %s=%r: %s", rejection.field, rejection.value, rejection.reason)
        return ProfileResult(site=context.site, out_dir=out_dir, profile=profile, rejected=rejected)

    def _digest(self, site: str, sections: list[Section], extract: ExtractManifest) -> str:
        # Shrink snippets until the whole digest fits the budget.
        snippet = self.config.snippet_words
        while True:
            text = _render_digest(site, sections, extract, snippet)
            if _words(text) <= self.config.max_digest_words or snippet <= 5:
                return text
            snippet = max(5, snippet // 2)

    def _select_entities(self, digest: str, candidates: list[Candidate], out_dir: Path) -> list[Entity]:
        if not candidates:
            return []
        schema = selection_schema([c.id for c in candidates])
        prompt = f"{digest}\n{render_candidates(candidates)}\n"

        def vote(index: int):
            return self.llm.parse(
                system=SELECTION_PROMPT,
                user=prompt,
                schema=schema,
                reasoning_effort=self.config.reasoning_effort,
                cache_salt=f"vote-{index}",
            )

        with ThreadPoolExecutor(max_workers=self.config.votes) as pool:
            selections = list(pool.map(vote, range(1, self.config.votes + 1)))

        entities, summary = aggregate(selections, candidates, min(self.config.min_votes, self.config.votes))
        (out_dir / VOTES).write_text(
            json.dumps(
                {
                    "selections": [s.model_dump(mode="json") for s in selections],
                    "summary": summary.model_dump(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        by_id = {c.id: c.text for c in candidates}
        for cid, n in summary.candidate_votes.items():
            if n < summary.min_votes:
                self.logger.info("  not enough votes (%d/%d): %s", n, summary.runs, by_id[cid])
        return entities

    def _ground(self, draft: ProfileDraft, digest: str) -> tuple[ProfileDraft, list[Rejection]]:
        """Keep only company fields the site text supports; everything removed is reported.

        Checks are deterministic (no LLM). The checklist is exempt: its questions
        are meant to go beyond the site and never become knowledge-base facts.
        """
        rejected: list[Rejection] = []
        if not is_grounded(draft.company_name, digest):
            rejected.append(Rejection(field="company_name", value=draft.company_name, reason="not in site text"))
            draft.company_name = ""
        if not is_grounded(draft.business_type, digest, BUSINESS_TYPE_RATIO):
            rejected.append(Rejection(field="business_type", value=draft.business_type, reason="not in site text"))
            draft.business_type = ""

        kept_locations = []
        for location in draft.locations:
            if is_grounded(location, digest):
                kept_locations.append(location)
            else:
                rejected.append(Rejection(field="locations", value=location, reason="not in site text"))
        draft.locations = kept_locations

        for item in draft.support_checklist:
            item.key = re.sub(r"[^a-z0-9]+", "_", item.key.lower()).strip("_")
        return draft, rejected


def _render_digest(site: str, sections: list[Section], extract: ExtractManifest, snippet_words: int) -> str:
    pages = {page.url: page for page in extract.pages}
    lines = [f"SITE: {site}", ""]

    current_page = None
    for section in sections:
        if section.page_url != current_page:
            current_page = section.page_url
            page = pages.get(section.page_url.split("#")[0])
            label = "SITE CHROME (navigation, footer)" if section.source == "boilerplate" else "PAGE"
            lines.append(f"{label}: {section.page_title or ''} <{section.page_url}>".rstrip())
            if page and page.description and section.source != "boilerplate":
                lines.append(f"  description: {page.description}")
        path = " > ".join(section.heading_path) or "(no heading)"
        lines.append(f"  [{section.id}] {path} :: {_opening(section.text, snippet_words)}")

    business = list({
        json.dumps(_compact(item), ensure_ascii=False, sort_keys=True): _compact(item)
        for item in _json_ld_items(extract) if _is_business(item)
    }.values())
    if business:
        lines += ["", "STRUCTURED DATA (JSON-LD):"]
        lines += [f"  {json.dumps(item, ensure_ascii=False)[:1500]}" for item in business]

    emails = sorted({e for page in extract.pages if page.is_usable for e in page.contacts.emails})
    phones = sorted({p for page in extract.pages if page.is_usable for p in page.contacts.phones})
    if emails or phones:
        lines += ["", f"CONTACT LINKS: emails={emails} phones={phones}"]
    return "\n".join(lines) + "\n"


def _opening(text: str, words: int) -> str:
    body = " ".join(line for line in text.splitlines() if not line.startswith("#"))
    tokens = body.split()
    return " ".join(tokens[:words]) + (" …" if len(tokens) > words else "")


def _json_ld_items(extract: ExtractManifest) -> list[dict]:
    items = []
    for page in extract.pages:
        if not page.is_usable:
            continue
        for block in page.json_ld:
            items.extend(block.get("@graph", [block]))
    return items


def _is_business(item: dict) -> bool:
    types = item.get("@type", [])
    return bool(BUSINESS_TYPES & set(types if isinstance(types, list) else [types]))


def _compact(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in JSON_LD_DROP}


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))
