import hashlib
import json
import re
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from kb_builder.llm import LLMClient

INDEX = "index.json"


class Chunk(BaseModel):
    id: str
    file: str
    title: str
    heading: str | None
    topic: str | None
    entity: str | None
    source_urls: list[str]
    text: str


class KnowledgeIndex:
    """Chunks of the knowledge-base files and their embeddings, searched by cosine similarity."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.chunks = chunks
        self.vectors = vectors

    @classmethod
    def load_or_build(
        cls, kb_dir: Path, index_dir: Path, llm: LLMClient, max_words: int = 300
    ) -> "KnowledgeIndex":
        """Reuse the saved index while the knowledge base is unchanged, else rebuild it."""
        fingerprint = _fingerprint(kb_dir, llm.settings.embedding_model, max_words)
        path = index_dir / INDEX
        if path.exists():
            data = json.loads(path.read_text())
            if data["fingerprint"] == fingerprint:
                return cls([Chunk(**c) for c in data["chunks"]], np.array(data["vectors"], dtype=np.float32))

        chunks = chunk_knowledge_base(kb_dir, max_words)
        if not chunks:
            raise FileNotFoundError(f"No knowledge-base files in {kb_dir}; run `kb build` first.")
        vectors = np.array(llm.embed([_embedding_text(c) for c in chunks]), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        index_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "chunks": [c.model_dump() for c in chunks],
                    "vectors": vectors.round(6).tolist(),
                },
                ensure_ascii=False,
            )
        )
        return cls(chunks, vectors)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        query = np.array(query_vector, dtype=np.float32)
        query /= np.linalg.norm(query)
        scores = self.vectors @ query
        best = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best]


def chunk_knowledge_base(kb_dir: Path, max_words: int) -> list[Chunk]:
    """One chunk per `##` section of every file; long sections split on paragraphs."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.rglob("*.md")):
        file = path.relative_to(kb_dir).as_posix()
        meta, body = _split_front_matter(path.read_text())
        title = meta.get("title") or path.stem
        for heading, text in _sections(body):
            for part, piece in enumerate(_split_long(text, max_words), start=1):
                anchor = _slug(heading or "intro")
                chunks.append(
                    Chunk(
                        id=f"{file}#{anchor}" + (f"-{part}" if part > 1 else ""),
                        file=file,
                        title=title,
                        heading=heading,
                        topic=meta.get("topic"),
                        entity=meta.get("entity"),
                        source_urls=meta.get("source_urls") or [],
                        text=piece,
                    )
                )
    return _unique_ids(chunks)


def _split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    header, _, body = text[4:].partition("\n---\n")
    meta = {}
    for line in header.splitlines():
        key, _, value = line.partition(": ")
        try:
            meta[key] = json.loads(value)
        except json.JSONDecodeError:
            meta[key] = value
    return meta, body


def _sections(body: str) -> list[tuple[str | None, str]]:
    """Split on `##` headings; the `#` title line is dropped (it lives in Chunk.title)."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in body.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    result = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if re.search(r"\w", text):
            result.append((heading, text))
    return result


def _split_long(text: str, max_words: int) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    parts, current = [], []
    for paragraph in re.split(r"\n\s*\n", text):
        if current and len(" ".join(current + [paragraph]).split()) > max_words:
            parts.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        parts.append("\n\n".join(current))
    return parts


def _embedding_text(chunk: Chunk) -> str:
    # Title and heading give a short chunk its subject ("220 PLN" alone means nothing).
    header = " > ".join(x for x in (chunk.title, chunk.heading) if x)
    return f"{header}\n{chunk.text}"


def _unique_ids(chunks: list[Chunk]) -> list[Chunk]:
    seen: dict[str, int] = {}
    for chunk in chunks:
        n = seen.get(chunk.id, 0) + 1
        seen[chunk.id] = n
        if n > 1:
            chunk.id = f"{chunk.id}-{n}"
    return chunks


def _fingerprint(kb_dir: Path, model: str, max_words: int) -> str:
    digest = hashlib.sha256(f"{model}|{max_words}".encode())
    for path in sorted(kb_dir.rglob("*.md")):
        digest.update(path.relative_to(kb_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "-", text.lower()).strip("-")[:40]
