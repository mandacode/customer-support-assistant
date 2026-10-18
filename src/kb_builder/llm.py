import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("kb_builder.llm")

# Pinned snapshots keep pipeline output (and RAGAS scores) reproducible.
DEFAULT_CHAT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Reasoning models spend output tokens on thinking; extraction tasks need little.
DEFAULT_REASONING_EFFORT = "low"


class LLMSettings(BaseModel):
    chat_model: str = DEFAULT_CHAT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    cache_dir: Path = Path("data/llm_cache")

    @classmethod
    def from_env(cls) -> "LLMSettings":
        load_dotenv()
        return cls(
            chat_model=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT) or None,
        )


class LLMClient:
    """OpenAI calls with structured output and an on-disk response cache.

    The cache key covers the model, prompts and output schema, so re-running a
    stage is free and deterministic until any of them changes.
    """

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        # Created lazily so fully cached runs work without an API key.
        if self._client is None:
            load_dotenv()
            self._client = OpenAI()
        return self._client

    def parse[T: BaseModel](self, *, system: str, user: str, schema: type[T]) -> T:
        model, effort = self.settings.chat_model, self.settings.reasoning_effort
        key = _hash(
            {"model": model, "effort": effort, "system": system, "user": user, "schema": schema.model_json_schema()}
        )
        cached = self._read("chat", key)
        if cached is not None:
            logger.info("[llm] cache hit  %s  %s", model, schema.__name__)
            return schema.model_validate(cached["output"])

        response = self.client.responses.parse(
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text_format=schema,
            **({"reasoning": {"effort": effort}} if effort else {}),
        )
        output = response.output_parsed
        if output is None:
            raise RuntimeError(f"{model} returned no parsable {schema.__name__}")

        usage = response.usage.model_dump() if response.usage else {}
        logger.info(
            "[llm] %s  %s  in=%s out=%s tokens",
            model, schema.__name__, usage.get("input_tokens"), usage.get("output_tokens"),
        )
        self._write("chat", key, {"model": model, "usage": usage, "output": output.model_dump(mode="json")})
        return output

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self.settings.embedding_model
        keys = [_hash({"model": model, "text": text}) for text in texts]
        vectors: dict[int, list[float]] = {}
        missing: list[int] = []
        for index, key in enumerate(keys):
            cached = self._read("embeddings", key)
            if cached is None:
                missing.append(index)
            else:
                vectors[index] = cached["vector"]

        if missing:
            # Send each distinct text once, even if it repeats in the batch.
            unique = list(dict.fromkeys(texts[i] for i in missing))
            response = self.client.embeddings.create(model=model, input=unique)
            fresh = {text: item.embedding for text, item in zip(unique, response.data)}
            for index in missing:
                vectors[index] = fresh[texts[index]]
                self._write("embeddings", keys[index], {"model": model, "vector": vectors[index]})
            logger.info("[llm] %s  embedded %d texts (%d cached)", model, len(unique), len(texts) - len(missing))
        return [vectors[i] for i in range(len(texts))]

    def _path(self, kind: str, key: str) -> Path:
        return self.settings.cache_dir / kind / key[:2] / f"{key}.json"

    def _read(self, kind: str, key: str) -> dict | None:
        path = self._path(kind, key)
        return json.loads(path.read_text()) if path.exists() else None

    def _write(self, kind: str, key: str, data: dict) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False))


def _hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
