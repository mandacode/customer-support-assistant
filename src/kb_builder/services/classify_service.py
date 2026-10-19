from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model

from kb_builder.context import SiteContext
from kb_builder.grounding import is_grounded
from kb_builder.llm import LLMClient
from kb_builder.models import (
    ClassifiedSection,
    ClassifyConfig,
    ClassifyResult,
    Rejection,
    Section,
    SiteProfile,
    Topic,
)
from kb_builder.prompts import CLASSIFY_PROMPT
from kb_builder.services.base_service import BaseService
from kb_builder.services.profile_service import PROFILE
from kb_builder.services.segment_service import SECTIONS, read_sections

CLASSIFIED = "classified.jsonl"
NO_ENTITY = "none"


class ClassifyService(BaseService[ClassifyResult]):
    """Give every section a topic and entities chosen from fixed lists."""

    name = "classify"

    def __init__(self, config: ClassifyConfig | None = None, llm: LLMClient | None = None) -> None:
        super().__init__()
        self.config = config or ClassifyConfig()
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def inputs(self, context: SiteContext) -> list[Path]:
        return [context.segmented_dir / SECTIONS, context.profile_dir / PROFILE]

    def run(self, context: SiteContext) -> ClassifyResult:
        sections = read_sections(context.segmented_dir / SECTIONS)
        profile = SiteProfile.model_validate_json((context.profile_dir / PROFILE).read_text())
        entity_ids = [e.id for e in profile.entities]
        entity_names = {e.id: e.name for e in profile.entities}

        classified: list[ClassifiedSection] = []
        rejected: list[Rejection] = []
        for start in range(0, len(sections), self.config.batch_size):
            batch = sections[start : start + self.config.batch_size]
            labels = self._classify(batch, profile, entity_ids)
            for section in batch:
                label = labels.get(section.id)
                if label is None:
                    rejected.append(Rejection(field="section", value=section.id, reason="no label returned"))
                    classified.append(ClassifiedSection(**section.model_dump()))
                    continue
                kept = []
                for entity_id in dict.fromkeys(label.entity_ids):
                    if entity_id == NO_ENTITY:
                        continue
                    # Deterministic check: the entity must actually be named in the section.
                    if is_grounded(entity_names[entity_id], section.text):
                        kept.append(entity_id)
                    else:
                        rejected.append(
                            Rejection(field="entity_ids", value=f"{section.id} -> {entity_id}", reason="not named in section")
                        )
                classified.append(
                    ClassifiedSection(**section.model_dump(), topic=label.topic, entity_ids=kept, reason=label.reason)
                )

        out_dir = context.classified_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / CLASSIFIED).open("w") as file:
            for section in classified:
                file.write(section.model_dump_json() + "\n")

        for section in classified:
            entities = ",".join(section.entity_ids) or "-"
            self.logger.info("  %-16s %-34s %s", section.topic or "?", entities[:34], section.id)
        return ClassifyResult(site=context.site, out_dir=out_dir, sections=classified, rejected=rejected)

    def _classify(self, batch: list[Section], profile: SiteProfile, entity_ids: list[str]) -> dict[str, BaseModel]:
        schema = _labels_schema([s.id for s in batch], entity_ids)
        result = self.llm.parse(
            system=CLASSIFY_PROMPT,
            user=self._prompt(batch, profile),
            schema=schema,
            reasoning_effort=self.config.reasoning_effort,
        )
        labels: dict[str, BaseModel] = {}
        for label in result.labels:
            labels.setdefault(label.section_id, label)
        return labels

    def _prompt(self, batch: list[Section], profile: SiteProfile) -> str:
        lines = [
            f"COMPANY: {profile.company_name} ({profile.business_type}), language: {profile.language}",
            "ENTITIES:",
            *[f"  {e.id}: {e.name} ({e.kind})" for e in profile.entities],
            "",
            "SECTIONS:",
        ]
        for section in batch:
            path = " > ".join(section.heading_path) or "(no heading)"
            source = " [site navigation/footer]" if section.source == "boilerplate" else ""
            lines += [f"--- section_id: {section.id}{source}", f"heading: {path}", _cut(section.text, self.config.max_section_words), ""]
        return "\n".join(lines)


def read_classified(path: Path) -> list[ClassifiedSection]:
    return [ClassifiedSection.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def _labels_schema(section_ids: list[str], entity_ids: list[str]) -> type[BaseModel]:
    """Output schema whose ids are enums, so the model can only pick existing ones."""
    section_id = Literal[tuple(section_ids)]  # type: ignore[valid-type]
    entity_id = Literal[tuple(entity_ids or [NO_ENTITY])]  # type: ignore[valid-type]
    label = create_model(
        "SectionLabel",
        section_id=(section_id, ...),
        topic=(Topic, ...),
        entity_ids=(list[entity_id], ...),
        reason=(str, Field(description="A few words, for human review.")),
    )
    return create_model("SectionLabels", labels=(list[label], ...))


def _cut(text: str, words: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:words]) + (" …" if len(tokens) > words else "")
