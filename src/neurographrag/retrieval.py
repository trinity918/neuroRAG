"""Retrievers (BM25, dense, graph) and the Community-Aware Cross-Lingual
Retrieval Fusion (C²RF) that combines them.

C²RF = Reciprocal Rank Fusion over the base retrievers, plus a *community prior*
that up-weights passages living in the knowledge-graph community most associated
with the query's concepts. This gives structural, global context to what would
otherwise be a purely rank-based fusion.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .embeddings import Embedder
from .ingestion import Passage
from .kg import KnowledgeGraph
from .ontology import Ontology
from .utils import tokenize


@dataclass
class Scored:
    passage_id: str
    score: float


@dataclass
class RetrievalResult:
    passage: Passage
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # retriever -> rank (1-based)
    community: int = -1


# --------------------------------------------------------------------------- #
# Base retrievers
# --------------------------------------------------------------------------- #
class BM25Retriever:
    name = "bm25"

    def __init__(self, cfg: Config, passages: list[Passage]):
        self.k1 = cfg.retrieval.bm25.k1
        self.b = cfg.retrieval.bm25.b
        self.lowercase = cfg.ingestion.lowercase
        self.ids = [p.id for p in passages]
        self.docs_tokens = [p.tokens for p in passages]
        self.doc_len = np.array([len(t) for t in self.docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.tf: list[Counter[str]] = [Counter(t) for t in self.docs_tokens]
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(passages)
        self.idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    def search(self, query: str, top_k: int) -> list[Scored]:
        q = tokenize(query, lowercase=self.lowercase)
        scores = np.zeros(len(self.ids), dtype=np.float32)
        for term in q:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return _topk(self.ids, scores, top_k)


class DenseRetriever:
    name = "dense"

    def __init__(self, cfg: Config, passages: list[Passage], embedder: Embedder):
        self.embedder = embedder
        self.ids = [p.id for p in passages]
        self.matrix = embedder.encode([p.searchable_text() for p in passages])
        self._faiss = None
        if cfg.embedding.backend == "sentence-transformers":
            self._try_faiss()

    def _try_faiss(self) -> None:  # pragma: no cover - only with faiss extra
        try:
            import faiss

            index = faiss.IndexFlatIP(self.matrix.shape[1])
            index.add(self.matrix)
            self._faiss = index
        except Exception:
            self._faiss = None

    def search(self, query: str, top_k: int) -> list[Scored]:
        q = self.embedder.encode([query])
        if q.shape[0] == 0:
            return []
        if self._faiss is not None:  # pragma: no cover
            k = min(top_k, len(self.ids))
            sims, idx = self._faiss.search(q, k)
            return [Scored(self.ids[j], float(sims[0][r])) for r, j in enumerate(idx[0]) if j >= 0]
        sims = (self.matrix @ q[0]).astype(np.float32)
        return _topk(self.ids, sims, top_k)


class GraphRetriever:
    """Concept-seeded graph expansion retriever."""

    name = "graph"

    def __init__(self, cfg: Config, ontology: Ontology, kg: KnowledgeGraph):
        self.cfg = cfg
        self.ontology = ontology
        self.kg = kg

    def seeds(self, query: str) -> set[str]:
        return self.ontology.match(query)

    def search(self, query: str, top_k: int) -> list[Scored]:
        seeds = self.seeds(query)
        if not seeds:
            return []
        dist = self.kg.expand(seeds, self.cfg.retrieval.graph.hops, self.cfg.retrieval.graph.max_expansion_nodes)
        # Concept relevance decays *sharply* with hop distance so that passages are
        # ranked overwhelmingly by how many actual query (seed) concepts they
        # contain. Expanded concepts give only mild supporting credit, which keeps
        # the graph retriever precise: a concept-dense passage that never mentions
        # the query concept can no longer outrank the on-topic one.
        hop_weight = {0: 2.0, 1: 0.3, 2: 0.1}
        concept_weight: dict[str, float] = {c: hop_weight.get(h, 0.05) for c, h in dist.items()}
        scores: dict[str, float] = defaultdict(float)
        for c, w in concept_weight.items():
            for pid in self.kg.concept_passages.get(c, []):
                scores[pid] += w
        ids = list(scores)
        arr = np.array([scores[i] for i in ids], dtype=np.float32)
        return _topk(ids, arr, top_k)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def _topk(ids: list[str], scores: np.ndarray, k: int) -> list[Scored]:
    if len(ids) == 0:
        return []
    k = min(k, len(ids))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [Scored(ids[i], float(scores[i])) for i in top if scores[i] > 0]


class HybridRetriever:
    """Holds the base retrievers and performs single / RRF / C²RF fusion."""

    def __init__(
        self,
        cfg: Config,
        ontology: Ontology,
        kg: KnowledgeGraph,
        passages: list[Passage],
        embedder: Embedder,
    ):
        self.cfg = cfg
        self.ontology = ontology
        self.kg = kg
        self.by_id = {p.id: p for p in passages}
        self.bm25 = BM25Retriever(cfg, passages)
        self.dense = DenseRetriever(cfg, passages, embedder)
        self.graph = GraphRetriever(cfg, ontology, kg)
        self._map = {"bm25": self.bm25, "dense": self.dense, "graph": self.graph}

    def _candidate_lists(self, query: str, retrievers: list[str]) -> dict[str, list[Scored]]:
        n = self.cfg.retrieval.candidates_per_retriever
        return {r: self._map[r].search(query, n) for r in retrievers}

    def _query_community_weights(self, query: str) -> tuple[set[str], dict[int, float]]:
        """Query concept-context (seeds + their neighbors) and the normalized
        weight of each knowledge-graph community for this query."""
        seeds = self.ontology.match(query)
        context = set(seeds)
        for c in seeds:
            context |= self.ontology.neighbors(c)
        votes: Counter[int] = Counter()
        for c in seeds:
            comm = self.kg.concept_community.get(c)
            if comm is not None:
                votes[comm] += 1
        total = sum(votes.values())
        weights = {comm: cnt / total for comm, cnt in votes.items()} if total else {}
        return context, weights

    def search(
        self,
        query: str,
        retrievers: list[str],
        fusion: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        top_k = top_k or self.cfg.retrieval.top_k
        lists = self._candidate_lists(query, retrievers)

        if fusion == "single":
            (only,) = retrievers
            ranked = lists[only]
            ranks = {only: {s.passage_id: i + 1 for i, s in enumerate(ranked)}}
            fused = {s.passage_id: s.score for s in ranked}
        else:
            fused, ranks = self._rrf(lists)
            if fusion == "c2rf":
                self._apply_community_prior(query, fused)

        results = self._materialize(fused, ranks, top_k)
        return results

    def _rrf(self, lists: dict[str, list[Scored]]):
        k = self.cfg.retrieval.fusion.rrf_k
        weights = self.cfg.retrieval.fusion.weights
        fused: dict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = {}
        for name, ranked in lists.items():
            w = weights.get(name, 1.0)
            ranks[name] = {}
            for i, s in enumerate(ranked):
                rank = i + 1
                ranks[name][s.passage_id] = rank
                fused[s.passage_id] += w * (1.0 / (k + rank))
        return fused, ranks

    def _apply_community_prior(self, query: str, fused: dict[str, float]) -> None:
        """C²RF: multiplicatively boost passages whose concepts overlap the
        query's concept-context *and* fall in the community most associated with
        the query. Gating on concept overlap (rather than a passage's fragile
        majority-vote community label) means a passage is boosted only when it
        actually mentions a query-relevant concept — so an on-topic passage is
        never demoted below a merely same-community distractor. The multiplicative
        (scale-free) form sharpens ordering without fabricating new top hits.
        """
        context, weights = self._query_community_weights(query)
        if not weights:
            return
        prior = self.cfg.retrieval.fusion.community_prior
        for pid in list(fused):
            concepts = self.kg.passage_concepts.get(pid, set())
            overlap = concepts & context
            if not overlap:
                continue
            # affinity = community-weight mass of the query-relevant concepts the
            # passage actually contains (capped at 1.0).
            aff = min(1.0, sum(weights.get(self.kg.concept_community.get(c, -1), 0.0) for c in overlap))
            if aff:
                fused[pid] *= 1.0 + prior * aff

    def _materialize(self, fused, ranks, top_k) -> list[RetrievalResult]:
        order = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[RetrievalResult] = []
        for pid, score in order:
            p = self.by_id.get(pid)
            if p is None:
                continue
            contrib = {name: rank[pid] for name, rank in ranks.items() if pid in rank}
            out.append(
                RetrievalResult(
                    passage=p,
                    score=float(score),
                    ranks=contrib,
                    community=self.kg.passage_community.get(pid, -1),
                )
            )
        return out
