"""Unit tests for the ranking metrics."""
import numpy as np

from neurographrag.evaluation import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["x"], {"x"}) == 1.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "z"}, 2) == 0.5
    assert recall_at_k(["a"], set(), 1) == 0.0


def test_average_precision_is_ordered():
    hi = average_precision(["a", "b", "x", "y"], {"a", "b"})
    lo = average_precision(["x", "y", "a", "b"], {"a", "b"})
    assert hi == 1.0
    assert hi > lo


def test_ndcg_perfect_and_bounds():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0
    val = ndcg_at_k(["x", "a"], {"a"}, 2)
    assert 0.0 < val < 1.0
    assert np.isclose(ndcg_at_k([], {"a"}, 5), 0.0)
