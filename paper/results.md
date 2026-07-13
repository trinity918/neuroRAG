# NeuroGraphRAG — Evaluation Results

- Embedding backend: `concept-hash`
- Queries: 24 across languages bn, en, hi, ta
- Seed: 20260713 · wall-clock: 1.301s

## Main results

| Configuration | MRR | MAP | Recall@5 | nDCG@10 | Faithfulness | Ans. Rel. |
|---|---|---|---|---|---|---|
| BM25 | 0.958 | 0.573 | 0.622 | 0.691 | 0.727 | 0.396 |
| Dense | 0.951 | 0.541 | 0.580 | 0.648 | 0.724 | 0.406 |
| Graph | 0.638 | 0.596 | 0.705 | 0.687 | 0.644 | 0.302 |
| RRF (BM25+Dense) | 1.000 | 0.589 | 0.608 | 0.707 | 0.718 | 0.407 |
| RRF (BM25+Dense+Graph) | 0.979 | 0.607 | 0.622 | 0.725 | 0.733 | 0.391 |
| **C²RF (ours)** | 0.979 | 0.648 | 0.712 | 0.766 | 0.729 | 0.390 |

![main results](figures/main.svg)

## Breakdown by query type

_Per-query-type ndcg@10 (rows = configuration)._

| Configuration | cross-lingual | factoid | global | multi-hop |
|---|---|---|---|---|
| BM25 | 0.582 | 0.761 | 0.632 | 0.832 |
| Dense | 0.670 | 0.655 | 0.561 | 0.754 |
| Graph | 0.678 | 0.799 | 0.345 | 0.777 |
| RRF (BM25+Dense) | 0.657 | 0.736 | 0.682 | 0.805 |
| RRF (BM25+Dense+Graph) | 0.715 | 0.758 | 0.617 | 0.832 |
| C²RF (ours) | 0.767 | 0.782 | 0.697 | 0.832 |
