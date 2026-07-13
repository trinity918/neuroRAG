# NeuroGraphRAG — Architecture

This document walks through the system component by component and explains the design decisions behind
the two novel pieces (ontology-grounded cross-lingual embeddings and C²RF fusion).

## Data flow

```
corpus.jsonl ─▶ ingestion ─▶ passages (+concept tags)
ontology.yaml ─┬▶ concept matcher ──────────────┐
               ├▶ knowledge graph (kg) ─▶ communities + summaries
               └▶ embedder (concept block)      │
passages ──────┴▶ embedder (char-ngram block)   │
                                                 ▼
                        HybridRetriever = BM25 ⊕ Dense ⊕ Graph  ── C²RF ─▶ ranked passages
                                                 ▼
                                generation (extractive | LLM)
                                                 ▼
                        evaluation (MRR/MAP/Recall/nDCG + RAGAS proxies)
```

## Modules

| Module | Responsibility | Key design choice |
|---|---|---|
| `ontology.py` | Load ontology; tag text with concept ids | One matcher, reused by KG + graph retriever + embedder — the *consistency* that makes everything align |
| `ingestion.py` | Load corpus, chunk, tag concepts | Title is indexed twice for short abstracts; unicode/Indic-aware tokens |
| `embeddings.py` | Text → vector | `concept-hash` = concept multi-hot ⊕ hashed char n-grams; `sentence-transformers` optional |
| `kg.py` | Build graph, detect communities, summarize | Ontology seed relations ⊕ corpus co-occurrence; weighted greedy modularity |
| `retrieval.py` | BM25/Dense/Graph + fusion | Seed-centric graph scoring; multiplicative, overlap-gated community prior |
| `generation.py` | Answer synthesis | Deterministic extractive default; Anthropic/OpenAI optional |
| `evaluation.py` | Metrics + harness | Document-level dedupe; offline RAGAS proxies |
| `pipeline.py` | Orchestration | Builds everything once; exposes `search`/`answer`/`graph_view` |
| `api.py` / `cli.py` | Serving + UX | FastAPI + argparse |

## Why the cross-lingual bridge works without a model

The `concept-hash` embedding is `[w·onehot(concepts) ; hashed_char_ngrams]`. Character n-grams do **not**
align across scripts (Latin vs Devanagari share no substrings), but the **concept block does**: the
ontology alias table maps `"N400"`, `"एन400"`, and `"এন৪০০"` to the same id `n400`, so their concept
vectors overlap. Cosine similarity between an English passage and its Hindi translation is therefore high
because they light up the same concept dimensions. This is why `Dense` and `C²RF` beat `BM25` on
cross-lingual queries (0.670 / 0.767 vs 0.582 nDCG@10) with no trained encoder.

For paraphrase beyond the ontology vocabulary, switch `embedding.backend` to `sentence-transformers`
(optionally with a LoRA adapter, see `scripts/train_lora.py`).

## Why C²RF needs concept-overlap gating

Community detection on a small graph is unstable, and a concept can cluster with a semantically different
neighbourhood (e.g., `alpha` clusters with the disorders it is "altered in"). A naive community prior
that boosts every passage whose *majority community label* matches the query will therefore demote the
genuinely on-topic passage. C²RF instead boosts a passage only by the community-weight mass of the
**query-relevant concepts it actually contains**, so:

- an on-topic passage (contains the query concept) is always eligible for the boost, and
- an off-topic same-community passage (does not contain any query-context concept) gets **zero** boost.

This is the fix validated by `tests/test_pipeline_and_retrieval.py::test_alpha_query_precision`.

## Extension points

- **Bigger corpus**: drop JSONL into `data/corpus/` with fields `{id, lang, title, text, source, year}`.
- **New concepts/languages**: extend `data/ontology/neuro_ontology.yaml` (add aliases under a language
  key); everything downstream picks them up.
- **Real dense encoder + LoRA**: `pip install -r requirements-extras.txt`, set
  `embedding.backend: sentence-transformers`, train with `scripts/train_lora.py`, point
  `embedding.lora_adapter` at the output.
- **LLM answers**: set `generation.provider` and the matching API key in `.env`.
- **Learned fusion**: replace `HybridRetriever._rrf` weights with a trained ranker.
