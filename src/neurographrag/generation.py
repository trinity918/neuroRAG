"""Answer synthesis over retrieved context.

Default provider is ``none`` — a deterministic *extractive* synthesizer that
needs no API key and makes the evaluation fully reproducible. ``anthropic`` and
``openai`` providers are available for grounded generative answers.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .config import Config
from .ontology import Ontology
from .retrieval import RetrievalResult

_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


@dataclass
class Answer:
    query: str
    text: str
    citations: list[str] = field(default_factory=list)         # passage ids
    contexts: list[RetrievalResult] = field(default_factory=list)
    community_summary: str | None = None
    provider: str = "none"


class Synthesizer:
    def __init__(self, cfg: Config, ontology: Ontology):
        self.cfg = cfg.generation
        self.ontology = ontology

    def answer(self, query: str, results: list[RetrievalResult], community_summary: str | None = None) -> Answer:
        ctx = results[: self.cfg.max_context_passages]
        summary = community_summary if self.cfg.include_community_summary else None
        if self.cfg.provider == "anthropic":
            text = self._anthropic(query, ctx, summary)
        elif self.cfg.provider == "openai":
            text = self._openai(query, ctx, summary)
        else:
            text = self._extractive(query, ctx)
        return Answer(
            query=query,
            text=text,
            citations=[r.passage.id for r in ctx],
            contexts=ctx,
            community_summary=summary,
            provider=self.cfg.provider,
        )

    # ---- extractive (default) ----
    def _extractive(self, query: str, ctx: list[RetrievalResult]) -> str:
        if not ctx:
            return "No relevant passage was retrieved for this query."
        q_concepts = self.ontology.match(query)
        best_sentences: list[str] = []
        for r in ctx[:2]:
            sentences = _SENT_SPLIT.split(r.passage.text.strip())
            scored = sorted(
                sentences,
                key=lambda s: len(self.ontology.match(s) & q_concepts),
                reverse=True,
            )
            if scored and scored[0]:
                best_sentences.append(scored[0].strip())
        # de-duplicate while preserving order
        seen: set[str] = set()
        uniq = [s for s in best_sentences if not (s in seen or seen.add(s))]
        return " ".join(uniq) if uniq else ctx[0].passage.text

    # ---- optional LLM providers ----
    def _prompt(self, query: str, ctx: list[RetrievalResult], summary: str | None) -> str:
        blocks = []
        if summary:
            blocks.append(f"[Community context] {summary}")
        for i, r in enumerate(ctx, 1):
            blocks.append(f"[{i}] ({r.passage.source}) {r.passage.text}")
        context = "\n".join(blocks)
        return (
            "You are a neuroscience research assistant. Answer the question using ONLY the "
            "context below. Cite sources as [n]. If the context is insufficient, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    def _anthropic(self, query, ctx, summary) -> str:  # pragma: no cover - needs network
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
            msg = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": self._prompt(query, ctx, summary)}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"[anthropic unavailable: {e}] " + self._extractive(query, ctx)

    def _openai(self, query, ctx, summary) -> str:  # pragma: no cover - needs network
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": self._prompt(query, ctx, summary)}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[openai unavailable: {e}] " + self._extractive(query, ctx)
