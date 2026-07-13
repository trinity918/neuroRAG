"""Corpus loading + passage chunking + concept tagging."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .ontology import Ontology
from .utils import read_jsonl, tokenize


@dataclass
class Passage:
    id: str            # "<doc_id>#p<k>"
    doc_id: str
    lang: str
    title: str
    source: str
    text: str
    tokens: list[str] = field(default_factory=list)
    concepts: set[str] = field(default_factory=set)
    year: int | None = None

    def searchable_text(self) -> str:
        # title carries strong signal for short abstracts; index it twice
        return f"{self.title}. {self.title}. {self.text}"


def _chunk_tokens(tokens: list[str], size: int, overlap: int) -> list[tuple[int, int]]:
    if len(tokens) <= size:
        return [(0, len(tokens))]
    spans: list[tuple[int, int]] = []
    step = max(1, size - overlap)
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        spans.append((start, end))
        if end == len(tokens):
            break
        start += step
    return spans


def load_passages(cfg: Config, ontology: Ontology) -> list[Passage]:
    """Read the corpus JSONL, split into passages, and tag each with concepts."""
    rows = read_jsonl(cfg.resolve(cfg.paths.corpus))
    lc = cfg.ingestion.lowercase
    passages: list[Passage] = []
    for row in rows:
        doc_id = row["id"]
        text = row["text"]
        title = row.get("title", "")
        # We tokenize on the raw text to compute chunk spans, but keep original
        # substrings for display by re-splitting on whitespace-ish boundaries.
        words = text.split()
        spans = _chunk_tokens(words, cfg.ingestion.chunk_size, cfg.ingestion.chunk_overlap)
        for k, (a, b) in enumerate(spans):
            chunk_text = " ".join(words[a:b])
            toks = tokenize(f"{title} {chunk_text}", lowercase=lc)
            if len(toks) < cfg.ingestion.min_chunk_tokens and len(spans) > 1:
                continue
            p = Passage(
                id=f"{doc_id}#p{k}",
                doc_id=doc_id,
                lang=row.get("lang", "en"),
                title=title,
                source=row.get("source", ""),
                text=chunk_text,
                tokens=toks,
                year=row.get("year"),
            )
            p.concepts = ontology.match(p.searchable_text())
            passages.append(p)
    return passages
