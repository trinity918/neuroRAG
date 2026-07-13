"""Typed configuration loaded from a YAML file (see configs/default.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    corpus: str
    ontology: str
    eval_queries: str
    artifacts: str = "artifacts"
    runs: str = "runs"


class Ingestion(BaseModel):
    chunk_size: int = 90
    chunk_overlap: int = 20
    min_chunk_tokens: int = 12
    lowercase: bool = True


class Embedding(BaseModel):
    backend: Literal["concept-hash", "sentence-transformers"] = "concept-hash"
    hash_dim: int = 2048
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    concept_weight: float = 2.0
    st_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    lora_adapter: str | None = None


class Community(BaseModel):
    algorithm: Literal["greedy_modularity", "label_propagation"] = "greedy_modularity"
    resolution: float = 1.0
    max_summary_passages: int = 4


class KG(BaseModel):
    min_concept_freq: int = 1
    cooccur_window: int = 1
    edge_min_weight: int = 1
    community: Community = Field(default_factory=Community)


class Fusion(BaseModel):
    method: Literal["c2rf", "rrf", "weighted"] = "c2rf"
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=lambda: {"bm25": 1.0, "dense": 1.0, "graph": 1.0})
    community_prior: float = 0.35


class GraphRetrieval(BaseModel):
    hops: int = 2
    max_expansion_nodes: int = 40


class BM25Params(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class Retrieval(BaseModel):
    top_k: int = 10
    candidates_per_retriever: int = 50
    fusion: Fusion = Field(default_factory=Fusion)
    graph: GraphRetrieval = Field(default_factory=GraphRetrieval)
    bm25: BM25Params = Field(default_factory=BM25Params)


class Generation(BaseModel):
    provider: Literal["none", "anthropic", "openai"] = "none"
    max_context_passages: int = 6
    include_community_summary: bool = True


class EvalConfigRow(BaseModel):
    name: str
    fusion: Literal["single", "rrf", "c2rf", "weighted"]
    retrievers: list[str]


class RagasCfg(BaseModel):
    enabled: bool = True
    metrics: list[str] = Field(default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision"])


class Evaluation(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    primary_metric: str = "ndcg@10"
    configs: list[EvalConfigRow] = Field(default_factory=list)
    ragas: RagasCfg = Field(default_factory=RagasCfg)


class Config(BaseModel):
    seed: int = 20260713
    paths: Paths
    ingestion: Ingestion = Field(default_factory=Ingestion)
    embedding: Embedding = Field(default_factory=Embedding)
    kg: KG = Field(default_factory=KG)
    retrieval: Retrieval = Field(default_factory=Retrieval)
    generation: Generation = Field(default_factory=Generation)
    evaluation: Evaluation = Field(default_factory=Evaluation)

    # Populated by load(): the repo root the relative paths resolve against.
    root: str = "."

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        # config lives in <root>/configs/<file>; root is two levels up.
        raw.setdefault("root", str(path.parent.parent))
        return cls.model_validate(raw)

    def resolve(self, relative: str) -> Path:
        """Resolve a config-relative path against the repository root."""
        return (Path(self.root) / relative).resolve()
