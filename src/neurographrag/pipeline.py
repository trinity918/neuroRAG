"""Top-level orchestrator wiring ontology -> passages -> KG -> retrieval -> answer."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import Config
from .embeddings import Embedder, build_embedder
from .evaluation import Evaluator
from .generation import Answer, Synthesizer
from .ingestion import Passage, load_passages
from .kg import KnowledgeGraph, build_graph
from .ontology import Ontology
from .retrieval import HybridRetriever, RetrievalResult
from .utils import set_seed


@dataclass
class BuildStats:
    passages: int
    concepts_in_graph: int
    edges: int
    communities: int
    languages: list[str]
    embedding_dim: int
    embedding_backend: str


class NeuroGraphRAG:
    def __init__(
        self,
        cfg: Config,
        ontology: Ontology,
        passages: list[Passage],
        kg: KnowledgeGraph,
        embedder: Embedder,
        retriever: HybridRetriever,
        synthesizer: Synthesizer,
    ):
        self.cfg = cfg
        self.ontology = ontology
        self.passages = passages
        self.kg = kg
        self.embedder = embedder
        self.retriever = retriever
        self.synth = synthesizer

    @classmethod
    def build(cls, cfg: Config) -> NeuroGraphRAG:
        set_seed(cfg.seed)
        ontology = Ontology.load(cfg.resolve(cfg.paths.ontology))
        passages = load_passages(cfg, ontology)
        kg = build_graph(cfg, ontology, passages)
        embedder = build_embedder(cfg, ontology)
        retriever = HybridRetriever(cfg, ontology, kg, passages, embedder)
        synthesizer = Synthesizer(cfg, ontology)
        return cls(cfg, ontology, passages, kg, embedder, retriever, synthesizer)

    # ---- public API ----
    def search(
        self,
        query: str,
        retrievers: list[str] | None = None,
        fusion: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        retrievers = retrievers or ["bm25", "dense", "graph"]
        fusion = fusion or self.cfg.retrieval.fusion.method
        return self.retriever.search(query, retrievers, fusion, top_k)

    def _dominant_community_summary(self, query: str) -> str | None:
        seeds = self.ontology.match(query)
        votes: Counter[int] = Counter()
        for c in seeds:
            comm = self.kg.concept_community.get(c)
            if comm is not None:
                votes[comm] += 1
        if not votes:
            return None
        comm_id = votes.most_common(1)[0][0]
        for c in self.kg.communities:
            if c.id == comm_id:
                return c.summary
        return None

    def answer(
        self,
        query: str,
        retrievers: list[str] | None = None,
        fusion: str | None = None,
        top_k: int | None = None,
    ) -> Answer:
        results = self.search(query, retrievers, fusion, top_k)
        summary = self._dominant_community_summary(query)
        return self.synth.answer(query, results, community_summary=summary)

    def evaluator(self) -> Evaluator:
        return Evaluator(self.cfg, self.retriever, self.embedder, self.synth)

    def graph_view(self, max_nodes: int = 200) -> dict:
        return self.kg.to_dict(max_nodes=max_nodes)

    def stats(self) -> BuildStats:
        return BuildStats(
            passages=len(self.passages),
            concepts_in_graph=self.kg.graph.number_of_nodes(),
            edges=self.kg.graph.number_of_edges(),
            communities=len(self.kg.communities),
            languages=sorted({p.lang for p in self.passages}),
            embedding_dim=self.embedder.dim,
            embedding_backend=self.embedder.name,
        )
