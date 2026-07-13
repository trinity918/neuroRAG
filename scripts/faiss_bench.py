"""Micro-benchmark: numpy brute-force cosine vs FAISS for the dense retriever.

Runs with the core install (numpy path only); if `faiss` is installed it also
times the FAISS path so you can see the crossover point as the corpus grows.

    python scripts/faiss_bench.py --n 20000 --dim 384 --queries 200
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def bench_numpy(mat: np.ndarray, queries: np.ndarray, k: int) -> float:
    t0 = time.time()
    for q in queries:
        sims = mat @ q
        np.argpartition(-sims, k)[:k]
    return time.time() - t0


def bench_faiss(mat: np.ndarray, queries: np.ndarray, k: int):
    try:
        import faiss
    except ImportError:
        return None
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    t0 = time.time()
    index.search(queries, k)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    mat = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    q = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    tn = bench_numpy(mat, q, args.k)
    print(f"numpy brute-force : {tn*1000:8.1f} ms for {args.queries} queries "
          f"over {args.n} vectors ({tn/args.queries*1e3:.2f} ms/query)")
    tf = bench_faiss(mat, q, args.k)
    if tf is None:
        print("faiss             : not installed (pip install faiss-cpu)")
    else:
        print(f"faiss IndexFlatIP : {tf*1000:8.1f} ms  (speedup {tn/tf:.1f}x)")


if __name__ == "__main__":
    main()
