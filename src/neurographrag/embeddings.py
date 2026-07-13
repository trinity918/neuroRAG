"""Embedding backends.

Two interchangeable encoders implement the same interface:

* ``ConceptHashEmbedder`` (default) — pure numpy. Concatenates an ontology
  *concept* vector with a hashed char-n-gram vector. The concept block is
  cross-lingual *by construction*: aliases in any language map to the same
  concept dimension, so parallel passages align even across scripts without any
  model download. This is the novel "ontology-grounded" representation.

* ``SentenceTransformerEmbedder`` (extras) — wraps a multilingual sentence
  encoder and, if configured, merges a LoRA/PEFT adapter for domain adaptation.

Both return L2-normalized rows so that dot product == cosine similarity.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Config
from .ontology import Ontology
from .utils import char_ngrams, l2_normalize, stable_hash, tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class ConceptHashEmbedder:
    """Ontology concept multi-hot ⊕ hashed char-n-gram bag-of-features."""

    name = "concept-hash"

    def __init__(self, cfg: Config, ontology: Ontology):
        self.cfg = cfg.embedding
        self.ontology = ontology
        self.concept_ix = ontology.concept_index()
        self.n_concepts = len(self.concept_ix)
        self.hash_dim = self.cfg.hash_dim
        self.dim = self.n_concepts + self.hash_dim
        self.lowercase = cfg.ingestion.lowercase

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        # --- concept block (cross-lingual bridge) ---
        for cid in self.ontology.match(text):
            j = self.concept_ix.get(cid)
            if j is not None:
                vec[j] += self.cfg.concept_weight
        # --- hashed char-n-gram block ---
        offset = self.n_concepts
        for tok in tokenize(text, lowercase=self.lowercase):
            for ng in char_ngrams(tok, self.cfg.char_ngram_min, self.cfg.char_ngram_max):
                vec[offset + stable_hash(ng, self.hash_dim)] += 1.0
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.vstack([self._encode_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)
        return l2_normalize(mat, axis=1)


class SentenceTransformerEmbedder:
    """Optional multilingual dense encoder with optional LoRA adapter merge."""

    name = "sentence-transformers"

    def __init__(self, cfg: Config, ontology: Ontology):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - exercised only with extras
            raise ImportError(
                "backend 'sentence-transformers' requires the 'st' extras: "
                "pip install -r requirements-extras.txt"
            ) from e
        self.model = SentenceTransformer(cfg.embedding.st_model)
        adapter = cfg.embedding.lora_adapter
        if adapter:
            self._merge_lora(adapter)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _merge_lora(self, adapter_path: str) -> None:  # pragma: no cover - extras path
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("lora_adapter set but PEFT is not installed (pip install peft).") from e
        transformer = self.model[0].auto_model
        merged = PeftModel.from_pretrained(transformer, adapter_path)
        self.model[0].auto_model = merged.merge_and_unload()

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - extras path
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)


def build_embedder(cfg: Config, ontology: Ontology) -> Embedder:
    if cfg.embedding.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg, ontology)
    return ConceptHashEmbedder(cfg, ontology)
