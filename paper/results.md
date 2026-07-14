# NeuroGraphRAG — Evaluation Results

- Embedding backend: `concept-hash`
- Queries: 72 across languages bn, en, hi, ta
- Seed: 20260713 · wall-clock: 3.576s

## Main results

| Configuration | MRR | MAP | Recall@5 | nDCG@10 | Faithfulness | Ans. Rel. |
|---|---|---|---|---|---|---|
| BM25 | 0.932 | 0.581 | 0.606 | 0.687 | 0.736 | 0.363 |
| Dense | 0.934 | 0.557 | 0.573 | 0.665 | 0.720 | 0.375 |
| Graph | 0.582 | 0.529 | 0.617 | 0.618 | 0.637 | 0.258 |
| RRF (BM25+Dense) | 0.954 | 0.585 | 0.601 | 0.691 | 0.724 | 0.370 |
| RRF (BM25+Dense+Graph) | 0.941 | 0.583 | 0.627 | 0.692 | 0.728 | 0.362 |
| **C²RF (ours)** | 0.941 | 0.607 | 0.633 | 0.722 | 0.725 | 0.363 |

![main results](figures/main.svg)

## Breakdown by query type

_Per-query-type ndcg@10 (rows = configuration)._

| Configuration | cross-lingual | factoid | global | multi-hop |
|---|---|---|---|---|
| BM25 | 0.678 | 0.763 | 0.550 | 0.692 |
| Dense | 0.668 | 0.730 | 0.494 | 0.712 |
| Graph | 0.624 | 0.753 | 0.304 | 0.625 |
| RRF (BM25+Dense) | 0.691 | 0.758 | 0.549 | 0.685 |
| RRF (BM25+Dense+Graph) | 0.698 | 0.755 | 0.557 | 0.643 |
| C²RF (ours) | 0.720 | 0.785 | 0.588 | 0.735 |
