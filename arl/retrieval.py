"""Small BM25 retriever over the policy knowledge base (markdown, chunked by '## ' sections)."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "is", "are", "be", "with", "can", "my", "i"}


def _tokens(text: str) -> list[str]:
    return [t.rstrip("s") for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str


def load_chunks(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        doc = path.read_text(encoding="utf-8")
        doc_title = doc.splitlines()[0].lstrip("# ").strip()
        for section in re.split(r"^## ", doc, flags=re.MULTILINE)[1:]:
            heading, _, body = section.partition("\n")
            chunks.append(Chunk(path.name, f"{doc_title} / {heading.strip()}", body.strip()))
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.docs = [_tokens(f"{c.title} {c.text}") for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 2) -> list[tuple[Chunk, float]]:
        q = _tokens(query)
        scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            tf = Counter(doc)
            score = sum(
                self.idf.get(t, 0) * tf[t] * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
                for t in q
            )
            if score > 0:
                scored.append((chunk, score))
        return sorted(scored, key=lambda x: -x[1])[:k]
