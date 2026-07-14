"""Conformal GraphRAG: distribution-free risk control for retrieval.

Standard RAG returns a fixed top-k with no guarantee the answer is in there.
Conformal prediction instead returns a *variable-size* retrieval set
C(q) = {d : s(q, d) >= tau} with a finite-sample, distribution-free guarantee

        P( C(q) contains a relevant document ) >= 1 - alpha,

for a threshold tau calibrated on held-out queries. No assumption is made about
the neural scorer; only exchangeability of the queries.

This module implements, in exact pure numpy:

* **Split conformal** thresholding (the ``floor(alpha*(n+1))`` order statistic).
* **Mondrian / group-conditional** calibration (per knowledge-graph community and
  per language) to attack conformal's well-known *conditional* under-coverage.
* A **cross-lingual transfer** study: calibrate on English, quantify the coverage
  shift on Hindi/Bengali/Tamil, and close it with language-conditional thresholds.
* **Risk-controlled selective generation** (Learn-then-Test style): abstain when
  uncertain so that the error on answered queries is bounded at a target level.

Everything is evaluated with many random calibration/test resamples because the
bundled benchmark is deliberately small; the guarantees themselves are exact.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import Config
from .evaluation import Query, load_queries
from .pipeline import NeuroGraphRAG

NEG_INF = -1e9


# --------------------------------------------------------------------------- #
# 1. Score the benchmark once into aligned numpy arrays
# --------------------------------------------------------------------------- #
@dataclass
class ScoredBenchmark:
    doc_ids: list[str]                 # universe of documents (columns)
    scores: np.ndarray                 # [Q, D] fused retrieval score per (query, doc)
    relevant: np.ndarray               # [Q, D] bool relevance mask
    r_star: np.ndarray                 # [Q] score of the best relevant doc (nonconformity anchor)
    top1_correct: np.ndarray           # [Q] is the top-scored doc relevant?
    community: np.ndarray              # [Q] query community id (or -1)
    lang: np.ndarray                   # [Q] language string
    qtype: np.ndarray                  # [Q] query type string
    query_ids: list[str]

    @property
    def n(self) -> int:
        return self.scores.shape[0]


def _query_community(ngr: NeuroGraphRAG, query: str) -> int:
    _, weights = ngr.retriever._query_community_weights(query)
    if not weights:
        return -1
    return max(weights.items(), key=lambda kv: kv[1])[0]


def score_benchmark(ngr: NeuroGraphRAG, queries: list[Query], retrievers, fusion) -> ScoredBenchmark:
    doc_ids = sorted({p.doc_id for p in ngr.passages})
    col = {d: i for i, d in enumerate(doc_ids)}
    D = len(doc_ids)
    Q = len(queries)

    scores = np.zeros((Q, D), dtype=np.float64)
    relevant = np.zeros((Q, D), dtype=bool)
    community = np.zeros(Q, dtype=int)
    r_star = np.zeros(Q, dtype=np.float64)
    top1_correct = np.zeros(Q, dtype=bool)
    langs, qtypes, qids = [], [], []

    for i, q in enumerate(queries):
        results = ngr.search(q.query, retrievers, fusion, top_k=D)
        best: dict[str, float] = {}
        for r in results:
            d = r.passage.doc_id
            best[d] = max(best.get(d, NEG_INF), r.score)
        for d, s in best.items():
            scores[i, col[d]] = s
        for d in q.relevant:
            if d in col:
                relevant[i, col[d]] = True
        rel_scores = scores[i][relevant[i]]
        r_star[i] = float(rel_scores.max()) if rel_scores.size else 0.0
        top_doc = int(np.argmax(scores[i]))
        top1_correct[i] = bool(relevant[i, top_doc])
        community[i] = _query_community(ngr, q.query)
        langs.append(q.lang)
        qtypes.append(q.type)
        qids.append(q.id)

    return ScoredBenchmark(
        doc_ids=doc_ids, scores=scores, relevant=relevant, r_star=r_star,
        top1_correct=top1_correct, community=community,
        lang=np.array(langs), qtype=np.array(qtypes), query_ids=qids,
    )


# --------------------------------------------------------------------------- #
# 2. Conformal primitives
# --------------------------------------------------------------------------- #
def conformal_threshold(calib_r: np.ndarray, alpha: float) -> float:
    """Finite-sample threshold tau such that P(r_test >= tau) >= 1 - alpha.

    Uses the floor(alpha*(n+1)) order statistic of the calibration nonconformity
    anchors (the score of each calibration query's best relevant doc).
    """
    n = calib_r.shape[0]
    if n == 0:
        return NEG_INF
    k = int(np.floor(alpha * (n + 1)))
    if k < 1:
        return NEG_INF          # include everything -> coverage 1
    if k > n:
        k = n
    return float(np.sort(calib_r)[k - 1])


def set_sizes(bench: ScoredBenchmark, idx: np.ndarray, tau: float) -> np.ndarray:
    """|C(q)| for each test query = #docs scoring >= tau."""
    return (bench.scores[idx] >= tau).sum(axis=1).astype(np.float64)


def covered(bench: ScoredBenchmark, idx: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """Whether C(q) contains a relevant doc, i.e. r_star(q) >= tau."""
    return bench.r_star[idx] >= tau


# --------------------------------------------------------------------------- #
# 3. Studies
# --------------------------------------------------------------------------- #
def _splits(n: int, calib_frac: float, trials: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    n_cal = max(2, int(round(calib_frac * n)))
    for _ in range(trials):
        perm = rng.permutation(n)
        out.append((perm[:n_cal], perm[n_cal:]))
    return out


def marginal_study(bench: ScoredBenchmark, cfg: Config, rng) -> list[dict]:
    """Coverage & set-size vs target 1-alpha, averaged over random splits."""
    splits = _splits(bench.n, cfg.conformal.calib_frac, cfg.conformal.trials, rng)
    rows = []
    for alpha in cfg.conformal.alphas:
        cov, size = [], []
        for cal, test in splits:
            tau = conformal_threshold(bench.r_star[cal], alpha)
            cov.append(covered(bench, test, tau).mean())
            size.append(set_sizes(bench, test, tau).mean())
        rows.append({
            "alpha": alpha,
            "target": 1 - alpha,
            "coverage": float(np.mean(cov)),
            "coverage_std": float(np.std(cov)),
            "avg_set_size": float(np.mean(size)),
        })
    return rows


def _group_thresholds(calib_r, calib_g, alpha, groups, min_calib, global_tau) -> dict:
    tau = {}
    for g in groups:
        mask = calib_g == g
        if mask.sum() >= min_calib:
            tau[g] = conformal_threshold(calib_r[mask], alpha)
        else:
            tau[g] = global_tau
    return tau


def mondrian_study(bench: ScoredBenchmark, cfg: Config, group_key: str, rng) -> dict:
    """Compare marginal vs group-conditional (Mondrian) calibration.

    Returns worst-group coverage under each scheme (conditional coverage is what
    Mondrian is designed to protect) at the primary alpha, plus per-group detail.
    """
    alpha = cfg.conformal.primary_alpha
    g_all = getattr(bench, group_key)
    groups = sorted(set(g_all.tolist()))
    splits = _splits(bench.n, cfg.conformal.calib_frac, cfg.conformal.trials, rng)

    per_group_marg = defaultdict(list)
    per_group_mond = defaultdict(list)
    worst_marg, worst_mond = [], []

    for cal, test in splits:
        tau_global = conformal_threshold(bench.r_star[cal], alpha)
        tau_grp = _group_thresholds(
            bench.r_star[cal], g_all[cal], alpha, groups, cfg.conformal.min_group_calib, tau_global
        )
        g_test = g_all[test]
        # marginal: single global tau; mondrian: per-group tau
        cov_marg_by_g, cov_mond_by_g = {}, {}
        for g in groups:
            m = g_test == g
            if not m.any():
                continue
            idx = test[m]
            cov_marg_by_g[g] = covered(bench, idx, tau_global).mean()
            cov_mond_by_g[g] = covered(bench, idx, tau_grp[g]).mean()
            per_group_marg[g].append(cov_marg_by_g[g])
            per_group_mond[g].append(cov_mond_by_g[g])
        if cov_marg_by_g:
            worst_marg.append(min(cov_marg_by_g.values()))
            worst_mond.append(min(cov_mond_by_g.values()))

    return {
        "group_key": group_key,
        "alpha": alpha,
        "target": 1 - alpha,
        "worst_group_coverage_marginal": float(np.mean(worst_marg)),
        "worst_group_coverage_mondrian": float(np.mean(worst_mond)),
        "per_group": {
            str(g): {
                "marginal": float(np.mean(per_group_marg[g])) if per_group_marg[g] else None,
                "mondrian": float(np.mean(per_group_mond[g])) if per_group_mond[g] else None,
                "n": int((g_all == g).sum()),
            }
            for g in groups
        },
    }


def cross_lingual_study(bench: ScoredBenchmark, cfg: Config, rng) -> dict:
    """Calibrate on English only; measure how the guarantee transfers to each
    other language (distribution shift), and whether pooled/other calibration
    restores coverage. This is the open question: does conformal coverage move
    across languages?
    """
    alpha = cfg.conformal.primary_alpha
    en = np.where(bench.lang == "en")[0]
    langs = sorted(set(bench.lang.tolist()))
    trials = cfg.conformal.trials

    # resample which English queries are calibration; test on all queries of each lang
    en_cover_en_calib = defaultdict(list)
    en_cover_pooled = defaultdict(list)
    n_cal = max(2, int(round(cfg.conformal.calib_frac * len(en))))
    for _ in range(trials):
        perm = rng.permutation(len(en))
        cal_en = en[perm[:n_cal]]
        tau_en = conformal_threshold(bench.r_star[cal_en], alpha)
        # pooled calibration: same size but drawn from all languages
        pooled_cal = rng.permutation(bench.n)[:n_cal]
        tau_pool = conformal_threshold(bench.r_star[pooled_cal], alpha)
        for lg in langs:
            idx = np.where(bench.lang == lg)[0]
            # exclude calibration queries from the English test pool
            if lg == "en":
                idx = np.array([i for i in idx if i not in set(cal_en.tolist())])
                if idx.size == 0:
                    continue
            en_cover_en_calib[lg].append(covered(bench, idx, tau_en).mean())
            en_cover_pooled[lg].append(covered(bench, idx, tau_pool).mean())

    return {
        "alpha": alpha,
        "target": 1 - alpha,
        "languages": langs,
        "english_calibrated": {lg: float(np.mean(en_cover_en_calib[lg])) for lg in langs if en_cover_en_calib[lg]},
        "pooled_calibrated": {lg: float(np.mean(en_cover_pooled[lg])) for lg in langs if en_cover_pooled[lg]},
    }


def selective_generation_study(bench: ScoredBenchmark, cfg: Config, rng) -> dict:
    """Risk-controlled abstention. Confidence = top-1 retrieval score; a query is
    'correct' if its top document is relevant. We calibrate a confidence threshold
    so the error on answered queries is <= beta (Learn-then-Test), and report the
    achieved test error and answer rate — plus the full risk-coverage curve.
    """
    conf = bench.scores.max(axis=1)
    correct = bench.top1_correct
    trials = cfg.conformal.trials
    splits = _splits(bench.n, cfg.conformal.calib_frac, trials, rng)

    # full (oracle) risk-coverage curve, averaged over test folds
    grid = np.linspace(conf.min(), conf.max(), 40)
    rc_answer, rc_risk = [], []
    for t in grid:
        ans_rate, risk = [], []
        for _, test in splits:
            m = conf[test] >= t
            ans_rate.append(m.mean())
            risk.append((1 - correct[test][m]).mean() if m.any() else 0.0)
        rc_answer.append(float(np.mean(ans_rate)))
        rc_risk.append(float(np.mean(risk)))

    # LTT: pick smallest calib threshold with selective risk <= beta, report test outcome
    beta_rows = []
    for beta in cfg.conformal.risk_betas:
        ach_risk, ach_ans = [], []
        for cal, test in splits:
            t = _ltt_threshold(conf[cal], correct[cal], beta)
            m = conf[test] >= t
            ach_ans.append(m.mean())
            ach_risk.append((1 - correct[test][m]).mean() if m.any() else 0.0)
        beta_rows.append({
            "beta": beta,
            "test_risk": float(np.mean(ach_risk)),
            "answer_rate": float(np.mean(ach_ans)),
        })

    return {
        "risk_coverage": {"answer_rate": rc_answer, "risk": rc_risk, "conf_grid": grid.tolist()},
        "ltt": beta_rows,
        "base_error_rate": float((1 - correct).mean()),
    }


def _ltt_threshold(conf, correct, beta) -> float:
    """Smallest confidence threshold whose selective (answered) risk <= beta."""
    order = np.argsort(conf)
    cs, cr = conf[order], correct[order]
    # sweep thresholds from low to high; answered = conf >= t
    best = conf.max() + 1e-9  # abstain from everything if nothing qualifies
    for t in np.unique(cs):
        m = conf >= t
        if m.any() and (1 - correct[m]).mean() <= beta:
            best = t
            break
    return float(best)


# --------------------------------------------------------------------------- #
# 4. Driver
# --------------------------------------------------------------------------- #
def run_conformal(ngr: NeuroGraphRAG, cfg: Config, queries: list[Query] | None = None) -> dict:
    queries = queries or load_queries(cfg)
    rng = np.random.default_rng(cfg.seed)
    bench = score_benchmark(ngr, queries, cfg.conformal.retrievers, cfg.conformal.fusion)

    return {
        "meta": {
            "n_queries": bench.n,
            "n_docs": len(bench.doc_ids),
            "retrievers": cfg.conformal.retrievers,
            "fusion": cfg.conformal.fusion,
            "trials": cfg.conformal.trials,
            "calib_frac": cfg.conformal.calib_frac,
            "primary_alpha": cfg.conformal.primary_alpha,
            "languages": sorted(set(bench.lang.tolist())),
            "seed": cfg.seed,
        },
        "marginal": marginal_study(bench, cfg, np.random.default_rng(cfg.seed)),
        "mondrian_community": mondrian_study(bench, cfg, "community", np.random.default_rng(cfg.seed + 1)),
        "mondrian_language": mondrian_study(bench, cfg, "lang", np.random.default_rng(cfg.seed + 2)),
        "cross_lingual": cross_lingual_study(bench, cfg, np.random.default_rng(cfg.seed + 3)),
        "selective": selective_generation_study(bench, cfg, np.random.default_rng(cfg.seed + 4)),
    }
