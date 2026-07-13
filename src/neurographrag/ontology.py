"""Domain ontology + multilingual concept matcher.

The matcher tags free text with ontology concept ids. Because aliases in every
language point to the same concept id, tagging is the mechanism that lets a
Hindi passage and its English counterpart share a representation downstream
(concept embeddings, graph nodes, community membership).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .utils import normalize, tokenize


@dataclass
class Concept:
    id: str
    canonical: str
    type: str
    aliases: dict[str, list[str]]  # lang -> surface forms
    definition: str = ""

    def all_aliases(self) -> list[str]:
        out: list[str] = [self.canonical]
        for forms in self.aliases.values():
            out.extend(forms)
        return out


@dataclass
class Relation:
    source: str
    target: str
    type: str


@dataclass
class Ontology:
    concepts: dict[str, Concept]
    relations: list[Relation]
    # matcher indexes (built in __post_init__-style factory below)
    _single: dict[str, str] = field(default_factory=dict)          # token -> concept id
    _phrases: list[tuple[str, str]] = field(default_factory=list)  # (normalized phrase, concept id)

    @classmethod
    def load(cls, path: str | Path) -> "Ontology":
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        concepts: dict[str, Concept] = {}
        for c in raw.get("concepts", []):
            concepts[c["id"]] = Concept(
                id=c["id"],
                canonical=c["canonical"],
                type=c.get("type", "Concept"),
                aliases=c.get("aliases", {}),
                definition=c.get("definition", ""),
            )
        relations = [
            Relation(source=r["source"], target=r["target"], type=r.get("type", "related_to"))
            for r in raw.get("relations", [])
            if r["source"] in concepts and r["target"] in concepts
        ]
        onto = cls(concepts=concepts, relations=relations)
        onto._build_index()
        return onto

    def _build_index(self) -> None:
        self._single.clear()
        self._phrases.clear()
        for cid, concept in self.concepts.items():
            for alias in concept.all_aliases():
                norm = normalize(alias)
                if not norm:
                    continue
                toks = tokenize(norm)
                if len(toks) == 1:
                    # single token: index the token itself (longest alias wins ties deterministically)
                    self._single.setdefault(toks[0], cid)
                else:
                    self._phrases.append((norm, cid))
        # match longer phrases first so "power spectral density" beats "density"
        self._phrases.sort(key=lambda p: len(p[0]), reverse=True)

    def match(self, text: str) -> set[str]:
        """Return the set of concept ids mentioned in ``text``."""
        found: set[str] = set()
        norm = normalize(text)
        # phrase matches on the normalized (spaced) string
        for phrase, cid in self._phrases:
            if phrase in norm:
                found.add(cid)
        # single-token matches on tokenized text
        for tok in tokenize(norm):
            cid = self._single.get(tok)
            if cid is not None:
                found.add(cid)
        return found

    def concept_index(self) -> dict[str, int]:
        """Stable concept-id -> column index mapping for concept embeddings."""
        return {cid: i for i, cid in enumerate(sorted(self.concepts))}

    def neighbors(self, cid: str) -> set[str]:
        out: set[str] = set()
        for r in self.relations:
            if r.source == cid:
                out.add(r.target)
            elif r.target == cid:
                out.add(r.source)
        return out

    def __len__(self) -> int:
        return len(self.concepts)
