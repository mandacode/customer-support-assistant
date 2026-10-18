import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from kb_builder.context import SiteContext
from kb_builder.grounding import fold
from kb_builder.models import (
    ClassifiedSection,
    DocumentGroup,
    DroppedSection,
    MergeConfig,
    MergeResult,
    SiteProfile,
    Topic,
)
from kb_builder.services.base_service import BaseService
from kb_builder.services.classify_service import CLASSIFIED, read_classified
from kb_builder.services.profile_service import PROFILE

DOCUMENTS = "documents.json"
PREVIEW = "preview"

# Site chrome (navigation, footer) is only worth keeping for contact details.
CONTACT_RE = re.compile(r"@|\+?\d[\d\s-]{6,}\d")


class MergeService(BaseService[MergeResult]):
    """Group classified sections into future knowledge-base files. No LLM.

    Rules, in order:
    - noise, unclassified and site-chrome sections without contact details are dropped;
    - offering sections go to `offerings/<entity>` (one file per entity named in
      them), or to `offerings/overview` when they name many entities at once;
    - all pricing sections go to one `pricing` file;
    - contact, company and process sections each form one file;
    - policy, domain knowledge and portfolio sections form one file per source page;
    - within a file, a section mostly contained in another one is a duplicate.
    """

    name = "merge"

    def __init__(self, config: MergeConfig | None = None) -> None:
        super().__init__()
        self.config = config or MergeConfig()

    def inputs(self, context: SiteContext) -> list[Path]:
        return [context.classified_dir / CLASSIFIED, context.profile_dir / PROFILE]

    def run(self, context: SiteContext) -> MergeResult:
        sections = read_classified(context.classified_dir / CLASSIFIED)
        profile = SiteProfile.model_validate_json((context.profile_dir / PROFILE).read_text())
        entity_names = {e.id: e.name for e in profile.entities}

        groups: dict[str, list[ClassifiedSection]] = {}
        dropped: list[DroppedSection] = []
        for section in sections:
            if section.source == "boilerplate" and not CONTACT_RE.search(section.text):
                dropped.append(DroppedSection(section_id=section.id, reason="site chrome"))
                continue
            keys = self._group_keys(section)
            if not keys:
                dropped.append(DroppedSection(section_id=section.id, reason=section.topic or "unclassified"))
            for key in keys:
                groups.setdefault(key, []).append(section)

        documents: list[DocumentGroup] = []
        for key, members in groups.items():
            kept, duplicates = self._dedupe(members)
            dropped.extend(duplicates)
            documents.append(self._document(key, kept, entity_names))

        documents.sort(key=lambda d: d.id)
        out_dir = context.merged_dir
        self._write(out_dir, documents, dropped, {s.id: s for s in sections})
        for document in documents:
            self.logger.info("  %-40s %d sections", document.id, len(document.section_ids))
        for item in dropped:
            note = f" (duplicate of {item.duplicate_of})" if item.duplicate_of else ""
            self.logger.info("  dropped %-14s %s%s", item.reason, item.section_id, note)
        return MergeResult(site=context.site, out_dir=out_dir, documents=documents, dropped=dropped)

    def _group_keys(self, section: ClassifiedSection) -> list[str]:
        match section.topic:
            case None | Topic.NOISE:
                return []
            case Topic.OFFERING:
                if not section.entity_ids:
                    return ["offerings/overview"]
                if len(section.entity_ids) >= self.config.overview_min_entities:
                    return ["offerings/overview"]
                return [f"offerings/{entity_id}" for entity_id in section.entity_ids]
            case Topic.PRICING:
                return ["pricing"]
            case Topic.CONTACT:
                return ["company/contact"]
            case Topic.COMPANY:
                return ["company/about"]
            case Topic.PROCESS:
                return ["process"]
            case Topic.POLICY:
                return [f"policies/{_page_slug(section.page_url)}"]
            case Topic.DOMAIN_KNOWLEDGE:
                return [f"domain-knowledge/{_page_slug(section.page_url)}"]
            case Topic.PORTFOLIO:
                return [f"portfolio/{_page_slug(section.page_url)}"]
        return []

    def _dedupe(self, members: list[ClassifiedSection]) -> tuple[list[ClassifiedSection], list[DroppedSection]]:
        """Drop sections mostly contained in a longer one (e.g. mobile copies of a block)."""
        kept: list[ClassifiedSection] = []
        dropped: list[DroppedSection] = []
        for section in members:
            duplicate = False
            for index, other in enumerate(kept):
                if self._is_duplicate(section.text, other.text):
                    if len(_words(section.text)) > len(_words(other.text)):
                        # The new one is the fuller copy: keep it in the earlier one's place.
                        dropped.append(DroppedSection(section_id=other.id, reason="duplicate", duplicate_of=section.id))
                        kept[index] = section
                    else:
                        dropped.append(DroppedSection(section_id=section.id, reason="duplicate", duplicate_of=other.id))
                    duplicate = True
                    break
            if not duplicate:
                kept.append(section)
        return kept, dropped

    def _is_duplicate(self, a: str, b: str) -> bool:
        """Same words AND the same numbers in the same order.

        Word overlap alone is not enough: two price tables share every word
        ("Felgi stalowe", "PLN") and differ only in their numbers.
        """
        (small, large) = (a, b) if len(_words(a)) <= len(_words(b)) else (b, a)
        small_words, large_words = _words(small), _words(large)
        if not small_words:
            return False
        if len(small_words & large_words) / len(small_words) < self.config.duplicate_containment:
            return False
        return _contains_run(_numbers(large), _numbers(small))

    def _document(self, key: str, members: list[ClassifiedSection], entity_names: dict[str, str]) -> DocumentGroup:
        entity_id = key.split("/", 1)[1] if key.startswith("offerings/") and key != "offerings/overview" else None
        # Final, human-readable titles in the site's language come from `normalize`.
        if entity_id:
            title = entity_names.get(entity_id, entity_id)
        elif key.startswith(("policies/", "domain-knowledge/", "portfolio/")):
            title = members[0].heading or _page_title(members[0].page_url)
        else:
            title = key
        return DocumentGroup(
            id=key,
            topic=members[0].topic,
            entity_id=entity_id,
            title=title,
            section_ids=[s.id for s in members],
            source_urls=list(dict.fromkeys(s.page_url for s in members)),
        )

    def _write(
        self,
        out_dir: Path,
        documents: list[DocumentGroup],
        dropped: list[DroppedSection],
        by_id: dict[str, ClassifiedSection],
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": [d.model_dump(mode="json") for d in documents],
            "dropped": [d.model_dump(mode="json") for d in dropped],
        }
        (out_dir / DOCUMENTS).write_text(json.dumps(data, indent=2, ensure_ascii=False))

        # Raw concatenated text per document, for review before `normalize`.
        preview_dir = out_dir / PREVIEW
        for document in documents:
            path = preview_dir / f"{document.id}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            parts = [f"<!-- {document.id} | {document.title} -->"]
            parts += [f"<!-- section: {sid} -->\n{by_id[sid].text}" for sid in document.section_ids]
            path.write_text("\n\n".join(parts) + "\n")


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", fold(text)))


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+", text)


def _contains_run(sequence: list[str], run: list[str]) -> bool:
    """True if `run` occurs in `sequence` as one contiguous block."""
    if not run:
        return True
    return any(sequence[i : i + len(run)] == run for i in range(len(sequence) - len(run) + 1))


def _page_slug(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    last = path.rsplit("/", 1)[-1] if path else ""
    last = re.sub(r"\.[a-z0-9]{1,5}$", "", last, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", fold(last)).strip("-") or "index"


def _page_title(url: str) -> str:
    return _page_slug(url).replace("-", " ").capitalize()
