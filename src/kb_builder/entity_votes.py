import re
from collections import Counter
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, create_model

from kb_builder.candidates import Candidate
from kb_builder.grounding import fold
from kb_builder.models import Entity, EntityKind


def selection_schema(candidate_ids: list[str]) -> type[BaseModel]:
    """Output schema whose ids are enums: the model can only pick listed candidates."""
    candidate_id = Literal[tuple(candidate_ids)]  # type: ignore[valid-type]
    group = create_model(
        "OfferingGroup",
        candidate_ids=(list[candidate_id], ...),
        name_candidate=(candidate_id, ...),
        kind=(EntityKind, ...),
    )
    return create_model("OfferingSelection", offerings=(list[group], ...))


def render_candidates(candidates: list[Candidate]) -> str:
    lines = ["CANDIDATES:"]
    for candidate in candidates:
        where = ", ".join(candidate.section_ids) or "structured data"
        lines.append(f"  {candidate.id}: {candidate.text}  ({', '.join(candidate.kinds)}; in {where})")
    return "\n".join(lines)


class VoteSummary(BaseModel):
    runs: int
    min_votes: int
    candidate_votes: dict[str, int]
    pair_votes: dict[str, int]


def aggregate(
    selections: list[BaseModel], candidates: list[Candidate], min_votes: int
) -> tuple[list[Entity], VoteSummary]:
    """Combine independent selections; keep only what enough of them agree on.

    - a candidate is an offering if at least `min_votes` runs selected it;
    - two such candidates form one entity if at least `min_votes` runs grouped them;
    - the entity's name is the candidate most often chosen as its name;
    - its kind is the kind most often given to groups containing that name.
    Entity names and sections therefore come from the site text, never from the model.
    """
    by_id = {c.id: c for c in candidates}
    groups_per_run = [
        [(set(group.candidate_ids) | {group.name_candidate}, group.name_candidate, group.kind) for group in run.offerings]
        for run in selections
    ]

    candidate_votes: Counter[str] = Counter()
    pair_votes: Counter[tuple[str, str]] = Counter()
    for groups in groups_per_run:
        candidate_votes.update({cid for ids, _, _ in groups for cid in ids})
        pair_votes.update({pair for ids, _, _ in groups for pair in combinations(sorted(ids), 2)})

    selected = sorted(c for c, n in candidate_votes.items() if n >= min_votes)
    parent = {c: c for c in selected}

    def root(c: str) -> str:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for (a, b), n in pair_votes.items():
        if n >= min_votes and a in parent and b in parent:
            parent[root(b)] = root(a)

    clusters: dict[str, list[str]] = {}
    for c in selected:
        clusters.setdefault(root(c), []).append(c)

    entities: list[Entity] = []
    for members in sorted(clusters.values(), key=lambda m: min(m)):
        name_votes = Counter(name for groups in groups_per_run for _, name, _ in groups if name in members)
        name_id = min(members, key=lambda c: (-name_votes[c], -candidate_votes[c], c))
        kind_votes = Counter(kind for groups in groups_per_run for ids, _, kind in groups if name_id in ids)
        kind = min(kind_votes, key=lambda k: (-kind_votes[k], list(EntityKind).index(k)))
        section_ids = list(dict.fromkeys(sid for c in members for sid in by_id[c].section_ids))
        entities.append(
            Entity(
                id="",
                name=by_id[name_id].text,
                kind=kind,
                candidate_ids=members,
                section_ids=section_ids,
            )
        )

    seen: set[str] = set()
    for entity in entities:
        base = _slug(entity.name) or "entity"
        slug, n = base, 2
        while slug in seen:
            slug, n = f"{base}-{n}", n + 1
        entity.id = slug
        seen.add(slug)

    summary = VoteSummary(
        runs=len(selections),
        min_votes=min_votes,
        candidate_votes=dict(sorted(candidate_votes.items())),
        pair_votes={f"{a}+{b}": n for (a, b), n in sorted(pair_votes.items())},
    )
    return entities, summary


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")[:40]
