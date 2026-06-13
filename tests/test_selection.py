"""Tests for matmercator.selection (q2_cv + select_gtm)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matmercator.config import PipelineConfig
from matmercator.selection import GTMSearchSpace
from matmercator.selection import q2_cv
from matmercator.selection import select_gtm


def _one_hot(assignments, k):
    R = np.zeros((len(assignments), k))
    R[np.arange(len(assignments)), assignments] = 1.0
    return R


def test_q2_cv_perfect_when_node_determines_property():
    """One-hot responsibilities + node-constant property => Q^2 ~ 1."""
    assign = [0] * 6 + [1] * 6
    R = _one_hot(assign, k=2)
    y = np.array([5.0] * 6 + [9.0] * 6)
    assert q2_cv(R, y, n_folds=3, seed=0) > 0.99


def test_q2_cv_near_zero_for_noise():
    """Random responsibilities vs an independent property => Q^2 not positive."""
    rng = np.random.default_rng(0)
    R = rng.random((80, 8))
    R /= R.sum(axis=1, keepdims=True)
    y = rng.standard_normal(80)
    assert q2_cv(R, y, n_folds=5, seed=0) < 0.3


def test_q2_cv_nan_without_variance():
    """A constant property has no variance to explain => NaN."""
    R = _one_hot([0, 0, 1, 1], k=2)
    assert np.isnan(q2_cv(R, np.full(4, 3.0), n_folds=2))


def test_select_gtm_returns_scored_grid():
    """select_gtm fits each candidate and returns a ranked, finite table."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.standard_normal((n, 6))
    df = pd.DataFrame(
        {
            "split": ["train"] * 60 + ["val"] * 20,
            "spacegroup.number": rng.integers(1, 231, n),
            "band_gap": rng.random(n),
            "formation_energy_per_atom": rng.standard_normal(n),
            "e_above_hull": rng.random(n),
        }
    )
    cfg = PipelineConfig(frame_set_size=40, gtm_niter=40, random_state=0)
    space = GTMSearchSpace(k=(6,), m=(2,), s=(0.3,), regul=(0.1,))
    res = select_gtm(cfg, df, X, space=space, n_folds=3)

    assert len(res["table"]) == 1
    assert res["best"]["mean_q2"] == res["table"][0]["mean_q2"]
    assert np.isfinite(res["best"]["mean_q2"])
    assert set(res["best"]["q2"]) == set(cfg.color_properties)
