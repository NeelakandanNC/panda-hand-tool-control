"""Embedding index over the tool catalog (stage 1 of the decision model: recall)."""

import hashlib
import json
from pathlib import Path

import numpy as np


def tool_text(tool: dict) -> str:
    """What gets embedded for a tool: its name in words plus its description."""
    return f"{tool['name'].replace('_', ' ')}: {tool['description']}"


class ToolIndex:
    def __init__(self, client, tools: list[dict], model: str, cache_dir: Path):
        self.client, self.tools, self.model = client, tools, model
        self.cache_path = cache_dir / f"embeddings-{model}.json"
        self.vectors = self._normalise(self._embed_cached([tool_text(t) for t in tools]))

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([item.embedding for item in response.data], dtype=float)

    def _embed_cached(self, texts: list[str]) -> np.ndarray:
        """Tool descriptions rarely change: only embed texts not already in the on-disk cache."""
        cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
        if missing:
            for (key, _), vec in zip(missing, self._embed([t for _, t in missing])):
                cache[key] = vec.tolist()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache))
        return np.array([cache[k] for k in keys])

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k tools by cosine similarity to the query: [(name, score), ...]."""
        q = self._normalise(self._embed([query])[0])
        scores = self.vectors @ q
        order = np.argsort(-scores)[:k]
        return [(self.tools[i]["name"], float(scores[i])) for i in order]
