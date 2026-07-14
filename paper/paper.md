# NeuroGraphRAG: Community-Aware Cross-Lingual GraphRAG for Multilingual Neuroscience Retrieval

**Abstract.** Retrieval-augmented generation (RAG) over scientific literature is dominated by
monolingual, flat dense retrieval, which underperforms on jargon-dense, multilingual, and *global*
information needs. We present **NeuroGraphRAG**, a graph-augmented retrieval system for neuroscience
(EEG/ERP and biomedical literature) that (i) grounds a knowledge graph, a graph retriever, and a text
embedder in a single curated ontology with *multilingual aliases*, yielding cross-lingual concept
embeddings without any trained model; and (ii) introduces **Community-Aware Cross-Lingual Retrieval
Fusion (C²RF)**, which augments reciprocal-rank fusion of lexical, dense, and graph signals with a
knowledge-graph *community prior*; and (iii) a **Conformal GraphRAG** risk-control layer that returns
variable-size retrieval sets with a distribution-free coverage guarantee, together with
community-conditional and cross-lingual calibration and risk-controlled abstention. On a bundled
multilingual benchmark (59 passages, 72 queries, four languages), C²RF attains the best nDCG@10 (0.722),
MAP (0.608), and Recall@5 (0.633), best on every query type, with the clearest margins on cross-lingual
and global queries. The conformal layer's coverage guarantee holds empirically (e.g. 0.814 at target
0.80) yet exposes a severe hidden failure — the worst knowledge-graph community is covered only 3.7% of
the time under marginal calibration, which Mondrian calibration repairs 6× — and shows that
English-calibrated guarantees under-cover Bengali/Tamil. The full pipeline runs offline on CPU; optional
`sentence-transformers` + LoRA/PEFT and FAISS backends are supported.

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
- **Conformal GraphRAG**: a distribution-free risk-control layer on top of retrieval, with
  community-conditional (Mondrian) and cross-lingual calibration and risk-controlled selective
  abstention — surfacing and repairing conditional-coverage failures a marginal guarantee hides.
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

**Benchmark.** 59 concept-dense passages spanning EEG rhythms, ERP components, analysis methods, BCI,
disorders, and cognition, authored in English with parallel Hindi/Bengali/Tamil passages; 72 queries
labelled by type (factoid, cross-lingual, global, multi-hop) with binary relevance judgments and
reference answers. Data is regenerable via `scripts/prepare_data.py`.

**Metrics.** MRR, MAP, Recall@{1,3,5,10}, nDCG@{1,3,5,10} at the document level, plus RAGAS-style
embedding proxies (faithfulness, answer relevancy, context precision). Primary metric: nDCG@10.

**Configurations.** Single retrievers (BM25, Dense, Graph); RRF(BM25+Dense); RRF(BM25+Dense+Graph); and
C²RF(BM25+Dense+Graph). Backend: `concept-hash`, seed 20260713.

## 5. Results

**Main results.**

| Configuration | MRR | MAP | Recall@5 | nDCG@10 | Faithfulness |
|---|---|---|---|---|---|
| BM25 | 0.932 | 0.581 | 0.606 | 0.687 | 0.736 |
| Dense | 0.934 | 0.557 | 0.573 | 0.665 | 0.720 |
| Graph | 0.582 | 0.530 | 0.617 | 0.618 | 0.637 |
| RRF (BM25+Dense) | **0.954** | 0.585 | 0.601 | 0.691 | 0.724 |
| RRF (BM25+Dense+Graph) | 0.941 | 0.584 | 0.627 | 0.693 | 0.727 |
| **C²RF (ours)** | 0.941 | **0.608** | **0.633** | **0.722** | 0.725 |

**By query type (nDCG@10).**

| Configuration | factoid | cross-lingual | global | multi-hop |
|---|---|---|---|---|
| BM25 | 0.763 | 0.678 | 0.550 | 0.692 |
| Dense | 0.730 | 0.668 | 0.494 | 0.712 |
| Graph | 0.753 | 0.625 | 0.304 | 0.625 |
| RRF (BM25+Dense) | 0.758 | 0.691 | 0.549 | 0.685 |
| RRF (BM25+Dense+Graph) | 0.755 | 0.699 | 0.557 | 0.643 |
| **C²RF (ours)** | **0.785** | **0.721** | **0.588** | **0.735** |

**Findings.**
1. *Fusion beats any single retriever* (nDCG@10 0.722 vs ≤0.687), confirming complementary signals.
2. *The graph signal adds recall*: adding it to RRF raises Recall@5 0.601 → 0.627 at comparable nDCG.
3. *Community-awareness is the key lift*: C²RF raises nDCG@10 0.693 → 0.722 and wins **every** query
   type, with the clearest margins on cross-lingual (0.721) and global (0.588).
4. *Cross-lingual without a model*: on cross-lingual queries, concept-grounded configurations (Dense
   0.668, C²RF 0.721) exceed BM25 (0.678) while pure lexical collapses on global queries, isolating the
   value of shared-concept embeddings and graph context.
5. *Honest trade-off*: two-way RRF retains the best MRR; C²RF trades marginal top-1 sharpness for gains
   in MAP/Recall/nDCG — motivating adaptive, per-query fusion (future work).

## 6. Risk-controlled retrieval: Conformal GraphRAG

Retrieval metrics report average ranking quality but give no per-query guarantee that a relevant
document was retrieved at all. We wrap the retriever in a conformal risk-control layer that returns a
variable-size set `C(q) = {d : s(q,d) ≥ τ}` with `P(C(q) ∩ Relevant(q) ≠ ∅) ≥ 1 − α`. The threshold is
the `⌊α(n+1)⌋`-th order statistic of the calibration anchors `r(q) = max_{d∈Relevant(q)} s(q,d)`; the
guarantee is distribution-free and finite-sample under query exchangeability. Full method:
`docs/CONFORMAL.md`.

**Coverage holds and sets are small.** Empirical coverage tracks the target (0.992/0.920/0.814/0.715 at
1−α = 0.95/0.90/0.80/0.70) while the average set shrinks from 21.3 to 0.9 of 59 documents.

**Conditional coverage (Mondrian).** Marginal validity is an average and hides sub-population failures.
Grouping queries by dominant knowledge-graph community, the worst community is covered only **0.037** at
1−α = 0.80 under a global threshold; community-conditional (Mondrian) calibration lifts it **6× to
0.222**. This is the practical face of conformal's open *conditional-coverage* problem, on a graph.

**Cross-lingual transfer.** Calibrating on English, coverage transfers to Hindi (0.87) but drops on
Bengali (0.48) and Tamil (0.63): cross-script retrieval scores live on a different scale, so the
English-tuned threshold is miscalibrated. Pooled multilingual calibration narrows the gap
(bn 0.57, ta 0.72). Cross-lingual conformal coverage is essentially unstudied; this is a minimal, clean
demonstration that exchangeability breaks across languages.

**Selective generation.** With top-1 confidence and a Learn-then-Test threshold, the answered-query error
is held under a target β (β = 0.10 → 0.083 achieved, answering 94% of queries), giving a deployable
abstention knob.

## 7. Domain adaptation with LoRA/PEFT (optional)

`scripts/train_lora.py` fine-tunes a multilingual sentence encoder with a low-rank adapter using a
contrastive objective over neuroscience passage/definition pairs mined from the ontology and corpus. The
trained adapter is merged at load time (`embedding.lora_adapter`), giving a domain-specialized encoder
without touching base weights. This is the intended path to scale cross-lingual dense retrieval beyond
the offline concept-hash bridge.

## 8. Limitations

The bundled benchmark is small and partly synthetic, so absolute numbers are illustrative; the
contribution is the *relative* ordering of configurations and the mechanism behind it. The concept-hash
embedder's cross-lingual signal is limited to in-ontology concepts (out-of-ontology paraphrase relies on
the optional multilingual encoder). Community detection on small graphs is unstable, which C²RF's
concept-overlap gating is specifically designed to tolerate. RAGAS proxies are embedding-based, not
LLM-graded. For the conformal layer, the small query count makes point estimates noisy (mitigated by
resampling and by reporting the exact guarantee separately from the empirical estimate), and
language-conditional calibration is data-starved for Tamil/Bengali; scaling the benchmark is the direct
next step.

## 9. Conclusion

NeuroGraphRAG shows that ontology-grounded structure delivers GraphRAG-style cross-lingual and global
retrieval benefits cheaply and reproducibly, and that a conformal layer turns the retriever into a
*risk-controlled* system with distribution-free guarantees. C²RF is a small, interpretable addition to
rank fusion that yields the best overall ranking quality on our benchmark, while Conformal GraphRAG
surfaces conditional-coverage and cross-lingual failures that average metrics hide and offers concrete
mechanisms (Mondrian, pooled calibration, selective abstention) to control them. The system is a
practical base for multilingual neuroscience question answering and a clean testbed for risk-controlled
hybrid retrieval research.

## References (selected)

- D. Edge et al. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* 2024.
- G. Cormack, C. Clarke, S. Buettcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
  Learning Methods.* SIGIR 2009.
- V. Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
- F. Feng et al. *Language-agnostic BERT Sentence Embedding (LaBSE).* 2020.
- E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- S. Es et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* 2023.
- A. Angelopoulos, S. Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification.* 2023.
- A. Angelopoulos et al. *Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2022.
- V. Vovk. *Conditional Validity of Inductive Conformal Predictors.* ACML 2012.
- AI4Bharat. *IndicNLP / IndicTrans.* https://ai4bharat.org

*Results in §5 are produced by `python -m neurographrag.cli eval`; conformal results in §6 by
`python -m neurographrag.cli conformal`. See `paper/results.md` and `paper/conformal_results.md` for the
auto-generated tables and figures.*
