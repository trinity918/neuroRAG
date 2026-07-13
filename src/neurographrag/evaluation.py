"""Reproducible evaluation: ranking metrics + RAGAS-style answer proxies.

Ranking metrics are computed at the *document* level: a ranked list of passages
is collapsed to unique parent documents (best rank kept) before scoring against
the relevance judgments. RAGAS proxies are embedding-based so they run offline;
install the ``ragas`` extra for the faithful, LLM-graded versions.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import Config
from .embeddings import Embedder
from .generation import Synthesizer
from .retrieval import HybridRetriever, RetrievalResult
from .utils import read_jsonl


# --------------------------------------------------------------------------- #
# Metric primitives (operate on ranked unique doc ids + a relevant set)
# --------------------------------------------------------------------------- #
def _dedupe_docs(ranked_passages: list[RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in ranked_passages:
        d = r.passage.doc_id
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# --------------------------------------------------------------------------- #
# Evaluation harness
# --------------------------------------------------------------------------- #
@dataclass
class Query:
    id: str
    lang: str
    type: str
    query: str
    relevant: set[str]
    answer: str


def load_queries(cfg: Config) -> list[Query]:
    rows = read_jsonl(cfg.resolve(cfg.paths.eval_queries))
    return [
        Query(
            id=r["id"],
            lang=r.get("lang", "en"),
            type=r.get("type", "factoid"),
            query=r["query"],
            relevant=set(r["relevant"]),
            answer=r.get("answer", ""),
        )
        for r in rows
    ]


class Evaluator:
    def __init__(
        self,
        cfg: Config,
        retriever: HybridRetriever,
        embedder: Embedder,
        synthesizer: Synthesizer,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.embedder = embedder
        self.synth = synthesizer
        self.k_values = cfg.evaluation.k_values
        self.max_k = max(self.k_values)

    def _ranking_metrics(self, ranked: list[str], relevant: set[str]) -> dict[str, float]:
        m: dict[str, float] = {
            "mrr": reciprocal_rank(ranked, relevant),
            "map": average_precision(ranked, relevant),
        }
        for k in self.k_values:
            m[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
            m[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
            m[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        return m

    def _ragas_proxy(self, q: Query, results: list[RetrievalResult]) -> dict[str, float]:
        ctx = results[: self.cfg.generation.max_context_passages]
        if not ctx:
            return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
        ans = self.synth.answer(q.query, results)
        texts = [ans.text, q.query, q.answer] + [r.passage.text for r in ctx]
        emb = self.embedder.encode(texts)
        a_vec, q_vec, ref_vec = emb[0], emb[1], emb[2]
        ctx_vecs = emb[3:]
        faithfulness = float(np.max(ctx_vecs @ a_vec)) if len(ctx_vecs) else 0.0
        answer_relevancy = float(a_vec @ q_vec)
        context_precision = float(np.mean(ctx_vecs @ ref_vec)) if q.answer else 0.0
        return {
            "faithfulness": max(0.0, faithfulness),
            "answer_relevancy": max(0.0, answer_relevancy),
            "context_precision": max(0.0, context_precision),
        }

    def run_config(self, name: str, retrievers: list[str], fusion: str, queries: list[Query]) -> dict:
        per_query = []
        agg: dict[str, list[float]] = defaultdict(list)
        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        ragas_agg: dict[str, list[float]] = defaultdict(list)

        for q in queries:
            results = self.retriever.search(q.query, retrievers, fusion, top_k=self.max_k)
            ranked = _dedupe_docs(results)
            rm = self._ranking_metrics(ranked, q.relevant)
            for key, val in rm.items():
                agg[key].append(val)
                by_type[q.type][key].append(val)
            row = {"id": q.id, "type": q.type, "lang": q.lang, **{k: round(v, 4) for k, v in rm.items()}}
            if self.cfg.evaluation.ragas.enabled:
                rp = self._ragas_proxy(q, results)
                for key, val in rp.items():
                    ragas_agg[key].append(val)
                row["ragas"] = {k: round(v, 4) for k, v in rp.items()}
            per_query.append(row)

        metrics = {k: float(np.mean(v)) for k, v in agg.items()}
        by_type_mean = {
            t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in by_type.items()
        }
        ragas = {k: float(np.mean(v)) for k, v in ragas_agg.items()} if ragas_agg else {}
        return {
            "name": name,
            "retrievers": retrievers,
            "fusion": fusion,
            "metrics": metrics,
            "by_type": by_type_mean,
            "ragas": ragas,
            "per_query": per_query,
        }

    def run(self, queries: list[Query] | None = None) -> dict:
        queries = queries or load_queries(self.cfg)
        t0 = time.time()
        rows = []
        for row in self.cfg.evaluation.configs:
            rows.append(self.run_config(row.name, row.retrievers, row.fusion, queries))
        return {
            "meta": {
                "seed": self.cfg.seed,
                "embedding_backend": self.cfg.embedding.backend,
                "num_queries": len(queries),
                "num_configs": len(rows),
                "primary_metric": self.cfg.evaluation.primary_metric,
                "elapsed_sec": round(time.time() - t0, 3),
                "languages": sorted({q.lang for q in queries}),
            },
            "results": rows,
        }
