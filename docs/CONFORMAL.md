# Conformal GraphRAG — Methodology

Retrieval metrics (nDCG, MRR) describe the *average* quality of a ranking. They are silent about the
question a practitioner actually cares about: *for this query, is a relevant document in what I
returned, and how sure can I be?* Conformal GraphRAG answers that with a distribution-free, finite-sample
guarantee, and then studies where the guarantee is fragile (across topics and across languages).

Implementation: [`src/neurographrag/conformal.py`](../src/neurographrag/conformal.py). Run:
`python -m neurographrag.cli conformal`.

## 1. Setup and nonconformity

For a query `q`, the retriever assigns every document a fused score `s(q, d)` (C²RF by default). A
prediction set is a super-level set

```
C(q; τ) = { d : s(q, d) ≥ τ }.
```

We want the smallest set that still contains a relevant document with high probability. Define the
**nonconformity anchor** of a query as the score of its *best* relevant document:

```
r(q) = max_{d ∈ Relevant(q)} s(q, d).
```

If `τ ≤ r(q)` then `C(q; τ)` contains at least one relevant document. So controlling coverage reduces to
controlling the event `r(q) ≥ τ`.

## 2. Split-conformal threshold

Given a calibration set of queries with anchors `r_1, …, r_n` (exchangeable with the test query), choose

```
τ = the ⌊α (n+1)⌋-th smallest of { r_i }      (τ = −∞ if ⌊α(n+1)⌋ < 1).
```

By the standard conformal argument, for a fresh test query,

```
P( r(q_test) ≥ τ )  ≥  1 − α,
```

i.e. **P( C(q_test) contains a relevant doc ) ≥ 1 − α**, with no assumption on the scorer. The set size
`|C(q; τ)|` is reported as the efficiency of the guarantee. Because the bundled benchmark is small, every
study repeats over many random calibration/test splits and reports the mean (and std) — the *guarantee*
is exact for each split; the *estimate* is what small `n` allows.

Empirically (72 queries, 59 docs, C²RF):

| target 1−α | empirical coverage | avg set size |
|---|---|---|
| 0.95 | 0.992 | 21.3 |
| 0.90 | 0.920 | 10.0 |
| 0.85 | 0.863 | 2.3 |
| 0.80 | 0.814 | 1.4 |
| 0.70 | 0.715 | 0.9 |

## 3. Conditional coverage (Mondrian)

Marginal validity is an *average* over queries; it can hide severe under-coverage on sub-populations.
Grouping queries by their dominant **knowledge-graph community**, we compute a separate threshold per
group (Mondrian conformal), falling back to the global threshold for groups with too few calibration
points (`min_group_calib`).

Result at 1−α = 0.80: the worst community's coverage is **0.037** under a single global threshold and
**0.222** under Mondrian — a 6× repair of a failure that the marginal number (0.81) completely masks.
This is the practical face of the open problem of *conditional* conformal coverage, instantiated on a
graph rather than on raw feature space.

## 4. Cross-lingual transfer

Does a guarantee calibrated in one language hold in another? We calibrate on English queries only and
measure coverage per test language:

| calibration | en | hi | bn | ta |
|---|---|---|---|---|
| English-only | 0.842 | 0.870 | 0.482 | 0.628 |
| pooled (multilingual) | 0.906 | 0.931 | 0.570 | 0.720 |

English calibration transfers well to Hindi (shared Latin acronyms + concept overlap) but **under-covers
Bengali and Tamil**, whose cross-script retrieval scores sit on a different scale — so the
English-tuned threshold is too aggressive. Pooling calibration data across languages narrows the gap.
Cross-lingual conformal coverage is, to our knowledge, essentially unstudied, and this is a clean minimal
demonstration that exchangeability breaks across languages.

## 5. Risk-controlled selective generation

Finally we let the system **abstain**. Confidence is the top-1 retrieval score; a query is "correct" if
its top document is relevant. Learn-then-Test picks the smallest confidence threshold whose *answered*
error is ≤ β on calibration, and we report the achieved test error and answer rate:

| target risk β | achieved test risk | answer rate |
|---|---|---|
| 0.10 | 0.083 | 0.940 |
| 0.20 | 0.096 | 0.993 |

The achieved risk respects the target while answering most queries — a usable knob for deployments where
a wrong grounded answer is costlier than an abstention.

## Limitations

- Small `n` (72 queries) makes point estimates noisy; we mitigate with resampling and report the exact
  guarantee separately from the empirical estimate.
- The nonconformity anchor uses binary relevance; graded relevance would refine set construction.
- Language-conditional calibration is data-starved for Tamil/Bengali here; scaling the benchmark (via
  `scripts/prepare_data.py`) is the direct next step.

## References

- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* 2005.
- Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction.* 2023.
- Angelopoulos et al. *Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2022.
- Vovk. *Conditional validity of inductive conformal predictors* (Mondrian). 2012.
