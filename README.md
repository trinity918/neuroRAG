<div align="center">

# 🧠 NeuroGraphRAG (Personalized by trinity918)

**Community-Aware Cross-Lingual GraphRAG for Multilingual Neuroscience Retrieval**

*Ontology-grounded knowledge graphs + hybrid BM25 / dense / graph retrieval for EEG, ERP, and biomedical literature — in English and Indic languages.*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Runs offline](https://img.shields.io/badge/runs-offline%2C%20no%20GPU-orange)](#quickstart)

</div>

---

## TL;DR

NeuroGraphRAG extends the GraphRAG paradigm to **multilingual neuroscience retrieval**. It builds a
neuroscience knowledge graph from a corpus of EEG/ERP abstracts, detects concept communities
(GraphRAG-style), and answers questions with a **hybrid retriever** that fuses lexical (BM25), dense
(multilingual embeddings), and structural (graph-expansion) signals. Its novel component,
**Community-Aware Cross-Lingual Retrieval Fusion (C²RF)**, adds a knowledge-graph *community prior* to
reciprocal-rank fusion and reaches the best nDCG@10, MAP, and Recall@5 on the bundled benchmark — and
the largest gains on exactly the queries GraphRAG is meant to help: **cross-lingual** and **global**.

On top of retrieval it adds **Conformal GraphRAG** — a risk-control layer that returns a variable-size
retrieval set with a *distribution-free, finite-sample guarantee* that a relevant document is inside it
with probability ≥ 1−α, plus **community-conditional** and **cross-lingual** calibration and
**risk-controlled selective abstention**. See [Risk-controlled retrieval](#risk-controlled-retrieval-conformal-graphrag).

The entire pipeline — ingest → knowledge graph → retrieval → conformal risk control → answer — **runs
offline on CPU with no model downloads**, thanks to an *ontology-grounded concept embedding* that is
cross-lingual by construction. Optional extras swap in `sentence-transformers` + **LoRA/PEFT** domain
adaptation and FAISS.

```bash
pip install -r requirements.txt
python -m neurographrag.cli demo          # build + evaluate + render results
python -m neurographrag.cli serve         # FastAPI backend + interactive UI at http://127.0.0.1:8000
```

---

## Why this is (a little) novel

Most RAG systems are monolingual and treat retrieval as a single dense lookup. GraphRAG adds global
structure but is English-centric and LLM-heavy. Neuroscience is an ideal stress test: dense in jargon,
multilingual (large Indic-language reader base), and full of *global* questions ("which ERP components
change in schizophrenia?") that a flat retriever answers poorly.

NeuroGraphRAG contributes four ideas, each backed by a number below:

1. **Ontology-grounded concept embeddings** — a curated neuroscience ontology with *multilingual
   aliases* is the single source of truth for the KG, the graph retriever, *and* the embedder. Because
   every language's alias maps to the same concept id, the concept block of the embedding is
   **cross-lingual without any trained model**. → biggest driver of the cross-lingual gain.
2. **Hybrid graph-augmented retrieval** — BM25 + dense + a precise, seed-centric graph-expansion
   retriever, fused with reciprocal rank fusion. → the graph signal is competitive with lexical+dense
   and adds recall (Recall@5 0.601 → 0.627).
3. **C²RF (Community-Aware Cross-Lingual Retrieval Fusion)** — a multiplicative *community prior* that
   promotes passages whose concepts fall in the knowledge-graph community most associated with the
   query. → lifts nDCG@10 0.693 → **0.722**, best on every query type.
4. **Conformal GraphRAG (risk control)** — distribution-free coverage guarantees on the retrieval set,
   with community-conditional (Mondrian) and cross-lingual calibration and risk-controlled abstention.
   → surfaces, and repairs, a **3.7% → 22% worst-community coverage** failure that marginal conformal
   hides. [Details below](#risk-controlled-retrieval-conformal-graphrag).

---

## Results

Bundled benchmark: **59 passages · 72 queries · 4 languages (en, hi, bn, ta)**, `concept-hash` backend,
seed `20260713`. Reproduce with `python -m neurographrag.cli eval`.

| Configuration | MRR | MAP | Recall@5 | **nDCG@10** | Faithfulness |
|---|---|---|---|---|---|
| BM25 | 0.932 | 0.581 | 0.606 | 0.687 | 0.736 |
| Dense (concept-hash) | 0.934 | 0.557 | 0.573 | 0.665 | 0.720 |
| Graph | 0.582 | 0.530 | 0.617 | 0.618 | 0.637 |
| RRF (BM25 + Dense) | **0.954** | 0.585 | 0.601 | 0.691 | 0.724 |
| RRF (BM25 + Dense + Graph) | 0.941 | 0.584 | 0.627 | 0.693 | 0.727 |
| **C²RF (ours)** | 0.941 | **0.608** | **0.633** | **0.722** | 0.725 |

**nDCG@10 by query type** — where the structure earns its keep:

| Configuration | factoid | cross-lingual | global | multi-hop |
|---|---|---|---|---|
| BM25 | 0.763 | 0.678 | 0.550 | 0.692 |
| Dense | 0.730 | 0.668 | 0.494 | 0.712 |
| RRF (BM25 + Dense) | 0.758 | 0.691 | 0.549 | 0.685 |
| **C²RF (ours)** | **0.785** | **0.721** | **0.588** | **0.735** |

C²RF wins overall and on **every query type**, with the clearest margins on **cross-lingual** and
**global** queries — exactly where knowledge-graph structure should help. Vanilla two-way RRF keeps the
best MRR (it occasionally has the sharper top-1), but C²RF trades a hair of top-1 for large gains in MAP,
Recall, and overall ranking quality. Full tables + figure: [`paper/results.md`](paper/results.md).

![results](paper/figures/main.svg)

---

## Risk-controlled retrieval (Conformal GraphRAG)

Retrieval metrics tell you the *average* quality of a ranking. They do **not** tell you, for a given
query, whether the answer is even in what you retrieved. Conformal GraphRAG wraps the retriever in a
distribution-free risk-control layer: instead of a fixed top-k, it returns a **variable-size set**
`C(q) = {d : score(q, d) ≥ τ}` with a finite-sample guarantee

> **P( C(q) contains a relevant document ) ≥ 1 − α**

for a threshold `τ` calibrated on held-out queries — no assumptions about the neural scorer, only query
exchangeability. Run it with `python -m neurographrag.cli conformal`.

**1. The guarantee holds, and sets stay small.** Empirical coverage tracks the target across α, while the
average set shrinks from the full corpus to ~1 document:

| target 1−α | empirical coverage | avg. set size (of 59 docs) |
|---|---|---|
| 0.95 | 0.992 | 21.3 |
| 0.90 | 0.920 | 10.0 |
| 0.80 | 0.814 | 1.4 |
| 0.70 | 0.715 | 0.9 |

**2. Marginal conformal hides brutal per-topic failures — and Mondrian repairs them.** At a 0.80 target,
the *worst knowledge-graph community* is covered only **3.7%** of the time under marginal calibration;
**community-conditional (Mondrian)** calibration lifts the worst group **6× to 22%**. Conditional
coverage is a known open problem in conformal prediction, made concrete here on a graph.

**3. Conformal guarantees do not transfer across languages for free.** Calibrated on English, coverage
drops on Bengali (0.48) and Tamil (0.63) versus English/Hindi (~0.85) — a genuine, and largely
unstudied, cross-lingual distribution-shift effect. Pooled multilingual calibration narrows the gap.

| calibration | en | hi | bn | ta |
|---|---|---|---|---|
| English-only | 0.842 | 0.870 | 0.482 | 0.628 |
| pooled (multilingual) | 0.906 | 0.931 | 0.570 | 0.720 |

**4. Risk-controlled selective generation.** Using top-1 confidence and a Learn-then-Test threshold, the
system abstains just enough to keep the error on *answered* queries under a target β (e.g. β = 0.10 →
0.083 achieved error while still answering 94% of queries).

Figures + full tables: [`paper/conformal_results.md`](paper/conformal_results.md) · methodology:
[`docs/CONFORMAL.md`](docs/CONFORMAL.md).

![reliability](paper/figures/conformal_reliability.svg)
![conditional coverage](paper/figures/conformal_community.svg)

> These numbers come from a deliberately small benchmark (72 queries) with heavy resampling; the
> *guarantees* are exact, the *estimates* are what a small n allows. Treat it as a rigorous reference
> implementation of risk-controlled multilingual GraphRAG, not a leaderboard.

---

## Quickstart

```bash
# 1. install the tiny core (numpy / scipy / networkx / pydantic / fastapi)
pip install -r requirements.txt

# 2. (re)generate the bundled multilingual benchmark
python scripts/prepare_data.py

# 3. one-shot demo: build KG + indexes, run the ablation grid, render tables + figure
python -m neurographrag.cli demo

# 3b. run the risk-controlled (conformal) study: coverage, Mondrian, cross-lingual, abstention
python -m neurographrag.cli conformal

# 4. ask a question (any supported language)
python -m neurographrag.cli query "Which ERP components are altered in schizophrenia?"
python -m neurographrag.cli query "अल्फा लय की आवृत्ति क्या है?"

# 5. launch the API + interactive knowledge-graph UI
python -m neurographrag.cli serve      # -> http://127.0.0.1:8000
```

On Windows without `make`, use the `python -m neurographrag.cli …` commands above; with Git Bash/WSL
the `Makefile` targets (`make demo`, `make api`, `make test`) also work.

### Optional accelerators (the "full-stack" paper configuration)

```bash
pip install -r requirements-extras.txt         # sentence-transformers, torch, peft, faiss, ...
# then flip the backend in configs/default.yaml:  embedding.backend: sentence-transformers
python scripts/train_lora.py --help            # LoRA/PEFT domain adaptation of the encoder
```

Everything degrades gracefully: no FAISS → numpy brute-force cosine; no `sentence-transformers` →
concept-hash embedder; no LLM key → deterministic extractive answers.

---

## Architecture

```
                      ┌──────────────────────────────────────────────┐
   multilingual       │  data/ontology/neuro_ontology.yaml            │
   corpus (jsonl) ──▶ │  concepts + multilingual aliases + relations  │◀── single source of truth
                      └───────────────┬──────────────────────────────┘
                                      │ concept matcher (cross-lingual)
         ┌────────────────────────────┼───────────────────────────────┐
         ▼                            ▼                                ▼
   ingestion/chunk            knowledge graph (kg)              embeddings
   + concept tagging     seed relations ⊕ co-occurrence     concept multi-hot ⊕
         │                community detection + summaries    hashed char n-grams
         │                            │                                │
         └──────────────┬─────────────┴───────────────┬────────────────┘
                        ▼                               ▼
              ┌───────────────────── HybridRetriever ─────────────────────┐
              │  BM25 (lexical)   Dense (semantic)   Graph (structural)    │
              │                   └──── C²RF fusion + community prior ──────┤
              └───────────────────────────┬───────────────────────────────┘
                                          ▼
                       generation (extractive | Anthropic | OpenAI)
                                          ▼
                    evaluation: MRR · MAP · Recall@k · nDCG@k · RAGAS proxies
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the component-by-component walkthrough.

---

## Repository layout

```
neurographrag/
├── src/neurographrag/        # the library
│   ├── ontology.py           # multilingual concept matcher (the cross-lingual bridge)
│   ├── ingestion.py          # corpus loading, chunking, concept tagging
│   ├── embeddings.py         # concept-hash (default) + sentence-transformers/LoRA backends
│   ├── kg.py                 # graph construction, community detection, summaries
│   ├── retrieval.py          # BM25 / dense / graph retrievers + C²RF fusion
│   ├── conformal.py          # Conformal GraphRAG: coverage guarantees, Mondrian, risk control
│   ├── conformal_report.py   # conformal figures + markdown report
│   ├── generation.py         # extractive / LLM answer synthesis
│   ├── evaluation.py         # ranking metrics + RAGAS-style proxies + harness
│   ├── pipeline.py           # orchestrator
│   ├── api.py                # FastAPI backend
│   └── cli.py                # index | query | eval | report | demo | serve
├── data/                     # ontology, corpus, eval queries (regenerate: scripts/prepare_data.py)
├── configs/default.yaml      # every knob, one file
├── web/                      # zero-build static UI (served by the API)
├── frontend/                 # optional Vite + React frontend
├── scripts/                  # prepare_data.py, train_lora.py, faiss_bench.py
├── paper/                    # paper draft (Markdown + LaTeX) + auto-generated results
├── tests/                    # pytest suite (runs on the core install)
└── docs/                     # architecture + design notes
```

---

## Reproducibility

- **Deterministic**: a single seed (`configs/default.yaml → seed`) controls all randomness; the
  concept-hash embedder is fully deterministic.
- **Self-contained**: the benchmark is authored in `scripts/prepare_data.py` and emitted to JSONL, so
  the data is regenerable and diff-able.
- **One command**: `python -m neurographrag.cli eval` writes a timestamped run to `runs/` plus
  `runs/latest.json`; `report` turns any run into `paper/results.md` + `paper/figures/main.svg`.
- **Tested**: `python -m pytest` covers tokenization, cross-lingual concept alignment, each retriever,
  the C²RF precision regression, ranking metrics, and the end-to-end ablation claim.

---

## Roadmap

- [ ] Scale the corpus with PubMed/OpenAlex neuroscience abstracts + AI4Bharat Indic translations.
- [ ] Contrastive LoRA fine-tuning of a multilingual encoder on mined neuroscience pairs (`train_lora.py`).
- [ ] Faithful RAGAS (LLM-graded) alongside the offline embedding proxies.
- [ ] Learned fusion weights (replace fixed RRF weights with a small ranker).
- [ ] Entity linking to UMLS / neuroscience ontologies (NIF, CogPO).

---

## Citation

```bibtex
@misc{neurographrag2026,
  title  = {NeuroGraphRAG: Community-Aware Cross-Lingual GraphRAG for Multilingual Neuroscience Retrieval},
  author = {NeuroGraphRAG contributors},
  year   = {2026},
  note   = {https://github.com/<your-org>/neurographrag}
}
```

## License

MIT — see [LICENSE](LICENSE).
