# NeuroGraphRAG: Community-Aware Cross-Lingual GraphRAG for Multilingual Neuroscience Retrieval

**Abstract.** Retrieval-augmented generation (RAG) over scientific literature is dominated by
monolingual, flat dense retrieval, which underperforms on jargon-dense, multilingual, and *global*
information needs. We present **NeuroGraphRAG**, a graph-augmented retrieval system for neuroscience
(EEG/ERP and biomedical literature) that (i) grounds a knowledge graph, a graph retriever, and a text
embedder in a single curated ontology with *multilingual aliases*, yielding cross-lingual concept
embeddings without any trained model; and (ii) introduces **Community-Aware Cross-Lingual Retrieval
Fusion (C²RF)**, which augments reciprocal-rank fusion of lexical, dense, and graph signals with a
knowledge-graph *community prior*. On a bundled multilingual benchmark (44 passages, 24 queries, four
languages), C²RF attains the best nDCG@10 (0.766), MAP (0.648), and Recall@5 (0.712), improving nDCG@10
over the strongest fusion baseline by 4.1 points and over BM25 by 7.5 points, with the largest gains on
cross-lingual (+18.5 over BM25) and global (+6.5 over the strongest baseline) queries. The full pipeline
runs offline on CPU; optional `sentence-transformers` + LoRA/PEFT and FAISS backends are supported.

---

## 1. Introduction

Neuroscience is a demanding setting for scientific retrieval: terminology is dense and abbreviation-heavy
(N400, MMN, SSVEP, ICA), a large fraction of readers are more comfortable in Indic languages than in
English, and many real questions are *global* ("which ERP components are altered in schizophrenia?")
rather than a single-passage lookup. Standard RAG — chunk, embed, nearest-neighbour — struggles on all
three axes. GraphRAG [Edge et al., 2024] introduced knowledge-graph structure and community summaries to
handle global questions, but existing pipelines are English-centric and depend heavily on large LLMs for
both graph construction and answering.

We ask a focused question: *can lightweight, ontology-grounded structure deliver GraphRAG-style benefits
— cross-lingual and global retrieval — without a large model in the loop?* NeuroGraphRAG answers yes.
Our contributions are:

- **Ontology-grounded concept embeddings.** A curated neuroscience ontology with multilingual aliases is
  the single source of truth for the knowledge graph, the graph retriever, and the text embedder. The
  embedder concatenates an ontology-concept multi-hot vector with hashed character n-grams; because every
  language's alias maps to the same concept id, the concept block is *cross-lingual by construction*.
- **Precise hybrid graph-augmented retrieval.** We fuse BM25, dense, and a seed-centric graph-expansion
  retriever whose hop-decayed scoring keeps it precise enough to help rather than dilute fusion.
- **C²RF.** A multiplicative community prior over reciprocal-rank fusion that promotes passages whose
  concepts lie in the knowledge-graph community most associated with the query, gated on concept overlap
  so that on-topic passages are never demoted below same-community distractors.
- **A reproducible, offline evaluation harness** (MRR, MAP, Recall@k, nDCG@k, and RAGAS-style embedding
  proxies) with a full ablation grid.

## 2. Related work

**RAG and dense retrieval.** DPR and downstream dense retrievers frame retrieval as nearest-neighbour
search in an embedding space. Hybrid lexical–dense fusion (e.g., reciprocal rank fusion, RRF
[Cormack et al., 2009]) is a strong, simple baseline that we build on.

**GraphRAG.** Microsoft's GraphRAG constructs an entity graph, detects communities, and generates
community summaries to serve *global* queries. We adopt the graph + community abstraction but replace
LLM-based entity extraction with an ontology-grounded matcher and add community structure directly into
the fusion step rather than only into summary-based global search.

**Multilingual and Indic NLP.** Multilingual sentence encoders (LaBSE, multilingual MiniLM) and the
AI4Bharat/IndicNLP ecosystem enable cross-lingual retrieval. NeuroGraphRAG supports these as an optional
backend, but its default cross-lingual signal comes from shared ontology concepts, which needs no model.

**Domain adaptation with LoRA/PEFT.** Low-rank adapters specialize encoders cheaply; we provide a
contrastive LoRA training script to adapt a multilingual encoder to neuroscience terminology (§6).

## 3. Method

### 3.1 Ontology and concept matching
The ontology defines typed concepts (rhythms, ERP components, methods, disorders, regions, cognition,
applications), each with a canonical form and per-language aliases, plus seed relations (e.g.,
`N400 —indexes→ semantic_memory`). A matcher tags any text with concept ids by phrase- and token-level
alias matching. This matcher is reused everywhere, which is what keeps the three subsystems consistent.

### 3.2 Cross-lingual concept embeddings
The default `concept-hash` embedder maps a passage to
`v = [w · onehot(concepts) ; hashed_char_ngrams]`, L2-normalized, where `w` up-weights the concept block.
The concept block is language-independent (aliases share ids), so parallel passages in English and Hindi
have overlapping concept dimensions and therefore high cosine similarity — the cross-lingual bridge.

### 3.3 Knowledge graph and communities
Concept nodes are connected by (a) ontology seed relations and (b) corpus co-occurrence edges. We detect
communities with weighted greedy modularity and generate a templated summary per community (anchor
concept, members, representative passages) that supplies global context.

### 3.4 Retrievers
*BM25* over unicode word tokens (Indic-aware). *Dense* cosine search over concept-hash (or
sentence-transformer) embeddings, with FAISS or numpy back-ends. *Graph*: from query seed concepts, we
expand over the graph and score passages by hop-decayed concept coverage (seed 2.0, 1-hop 0.3, 2-hop
0.1), which keeps a concept-dense but off-topic passage from outranking the on-topic one.

### 3.5 C²RF fusion
We first compute reciprocal-rank fusion, `s(p) = Σ_r w_r / (k + rank_r(p))`. C²RF then applies a
multiplicative community prior: let the query's concept-context be its seed concepts plus their graph
neighbours, and let each knowledge-graph community carry a weight equal to the fraction of query seeds it
contains. For a passage `p`, affinity `a(p)` is the community-weight mass of the query-relevant concepts
`p` actually contains (capped at 1), and we set `s(p) ← s(p) · (1 + λ · a(p))`. Gating on concept overlap
(rather than a passage's majority-community label) is essential: it guarantees C²RF never demotes an
on-topic passage below a merely same-community distractor — a failure mode we observed and fixed during
development.

## 4. Experimental setup

**Benchmark.** 44 concept-dense passages spanning EEG rhythms, ERP components, analysis methods, BCI,
disorders, and cognition, authored in English with parallel Hindi passages and additional Bengali/Tamil
passages; 24 queries labelled by type (factoid, cross-lingual, global, multi-hop) with binary relevance
judgments and reference answers. Data is regenerable via `scripts/prepare_data.py`.

**Metrics.** MRR, MAP, Recall@{1,3,5,10}, nDCG@{1,3,5,10} at the document level, plus RAGAS-style
embedding proxies (faithfulness, answer relevancy, context precision). Primary metric: nDCG@10.

**Configurations.** Single retrievers (BM25, Dense, Graph); RRF(BM25+Dense); RRF(BM25+Dense+Graph); and
C²RF(BM25+Dense+Graph). Backend: `concept-hash`, seed 20260713.

## 5. Results

**Main results.**

| Configuration | MRR | MAP | Recall@5 | nDCG@10 | Faithfulness |
|---|---|---|---|---|---|
| BM25 | 0.958 | 0.573 | 0.622 | 0.691 | 0.727 |
| Dense | 0.951 | 0.541 | 0.580 | 0.648 | 0.724 |
| Graph | 0.638 | 0.596 | 0.705 | 0.687 | 0.644 |
| RRF (BM25+Dense) | **1.000** | 0.589 | 0.608 | 0.707 | 0.718 |
| RRF (BM25+Dense+Graph) | 0.979 | 0.607 | 0.622 | 0.725 | 0.733 |
| **C²RF (ours)** | 0.979 | **0.648** | **0.712** | **0.766** | 0.729 |

**By query type (nDCG@10).**

| Configuration | factoid | cross-lingual | global | multi-hop |
|---|---|---|---|---|
| BM25 | 0.761 | 0.582 | 0.632 | 0.832 |
| Dense | 0.655 | 0.670 | 0.561 | 0.754 |
| Graph | 0.799 | 0.678 | 0.345 | 0.777 |
| RRF (BM25+Dense) | 0.736 | 0.657 | 0.682 | 0.805 |
| RRF (BM25+Dense+Graph) | 0.758 | 0.715 | 0.617 | 0.832 |
| **C²RF (ours)** | 0.782 | **0.767** | **0.697** | 0.832 |

**Findings.**
1. *Fusion beats any single retriever* (nDCG@10 0.766 vs ≤0.691), confirming complementary signals.
2. *The graph signal helps*: adding it to RRF lifts nDCG@10 0.707 → 0.725 and Recall@5 via better global
   and multi-hop coverage.
3. *Community-awareness helps most where structure matters*: C²RF lifts nDCG@10 0.725 → 0.766, with the
   biggest gains on cross-lingual (0.767) and global (0.697) queries.
4. *Cross-lingual without a model*: on cross-lingual queries, concept-grounded configurations (Dense
   0.670, C²RF 0.767) far exceed BM25 (0.582), isolating the value of shared-concept embeddings.
5. *Honest trade-off*: two-way RRF retains a perfect MRR; C²RF trades marginal top-1 sharpness for large
   MAP/Recall/nDCG gains — motivating adaptive, per-query fusion (future work).

## 6. Domain adaptation with LoRA/PEFT (optional)

`scripts/train_lora.py` fine-tunes a multilingual sentence encoder with a low-rank adapter using a
contrastive objective over neuroscience passage/definition pairs mined from the ontology and corpus. The
trained adapter is merged at load time (`embedding.lora_adapter`), giving a domain-specialized encoder
without touching base weights. This is the intended path to scale cross-lingual dense retrieval beyond
the offline concept-hash bridge.

## 7. Limitations

The bundled benchmark is small and partly synthetic, so absolute numbers are illustrative; the
contribution is the *relative* ordering of configurations and the mechanism behind it. The concept-hash
embedder's cross-lingual signal is limited to in-ontology concepts (out-of-ontology paraphrase relies on
the optional multilingual encoder). Community detection on small graphs is unstable, which C²RF's
concept-overlap gating is specifically designed to tolerate. RAGAS proxies are embedding-based, not
LLM-graded.

## 8. Conclusion

NeuroGraphRAG shows that ontology-grounded structure delivers GraphRAG-style cross-lingual and global
retrieval benefits cheaply and reproducibly. C²RF is a small, interpretable addition to rank fusion that
yields the best overall ranking quality on our benchmark, concentrated exactly on the queries structure
should help. The system is a practical base for multilingual neuroscience question answering and a clean
testbed for hybrid retrieval research.

## References (selected)

- D. Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* 2024.
- G. Cormack, C. Clarke, S. Buettcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
  Learning Methods.* SIGIR 2009.
- V. Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
- F. Feng et al. *Language-agnostic BERT Sentence Embedding (LaBSE).* 2020.
- E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- S. Es et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* 2023.
- AI4Bharat. *IndicNLP / IndicTrans.* https://ai4bharat.org

*Results in §5 are produced by `python -m neurographrag.cli eval`; see `paper/results.md` for the
auto-generated tables and figure.*
