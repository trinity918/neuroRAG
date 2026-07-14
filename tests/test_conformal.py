"""Conformal retrieval: threshold correctness + empirical coverage guarantees."""
import numpy as np

from neurographrag.conformal import conformal_threshold, run_conformal


def test_threshold_is_the_right_order_statistic():
    calib = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # k = floor(alpha*(n+1)) = floor(0.2*11) = 2  -> 2nd smallest = 0.2
    assert conformal_threshold(calib, 0.2) == 0.2
    # tiny alpha -> include everything
    assert conformal_threshold(calib, 0.001) < 0


def test_coverage_guarantee_holds_on_exchangeable_data():
    """The core promise: over random splits, empirical coverage >= 1 - alpha."""
    rng = np.random.default_rng(0)
    alpha = 0.2
    covs = []
    for _ in range(2000):
        r = rng.normal(size=40)             # exchangeable nonconformity anchors
        cal, test = r[:20], r[20:]
        tau = conformal_threshold(cal, alpha)
        covs.append((test >= tau).mean())
    # conformal is marginally valid: mean coverage >= 1 - alpha (small MC slack)
    assert np.mean(covs) >= (1 - alpha) - 0.02


def _small_run(ngr):
    cfg = ngr.cfg.model_copy(deep=True)
    cfg.conformal.trials = 120
    return run_conformal(ngr, cfg)


def test_marginal_coverage_tracks_target(ngr):
    run = _small_run(ngr)
    for row in run["marginal"]:
        # empirical coverage should not fall far below the target (small-n slack)
        assert row["coverage"] >= row["target"] - 0.08


def test_set_size_shrinks_as_target_drops(ngr):
    run = _small_run(ngr)
    sizes = [r["avg_set_size"] for r in run["marginal"]]  # targets decrease down the list
    assert all(sizes[i] >= sizes[i + 1] - 1e-9 for i in range(len(sizes) - 1))


def test_mondrian_helps_worst_community(ngr):
    run = _small_run(ngr)
    mc = run["mondrian_community"]
    # community-conditional calibration must not hurt, and here it strictly helps
    assert mc["worst_group_coverage_mondrian"] >= mc["worst_group_coverage_marginal"]


def test_selective_generation_controls_risk(ngr):
    run = _small_run(ngr)
    for row in run["selective"]["ltt"]:
        # achieved test risk should respect the target (allow small finite-sample slack)
        assert row["test_risk"] <= row["beta"] + 0.05
