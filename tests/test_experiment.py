"""Tests for matmercator.experiment helpers (q2_knn, standardize_join)."""

from __future__ import annotations

import numpy as np

from matmercator.experiment import q2_knn
from matmercator.experiment import standardize_join


def test_q2_knn_high_for_clustered_signal():
    """k-NN on a 2-D embedding predicts a cluster-aligned label well."""
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.05, (40, 2)), rng.normal(5, 0.05, (40, 2))]
    )
    y = np.concatenate([np.zeros(40), np.ones(40)])
    assert q2_knn(coords, y, n_neighbors=5, n_folds=4, seed=0) > 0.9


def test_q2_knn_low_for_noise():
    """Random coordinates vs an independent label give non-positive Q^2."""
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((100, 2))
    y = rng.standard_normal(100)
    assert q2_knn(coords, y, n_folds=5, seed=0) < 0.3


def test_standardize_join_shape_and_unit_scale():
    """Single block is z-scored to ~unit std on the frame rows."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 5, (50, 3))
    b = rng.normal(2, 1, (50, 7))
    frame = np.arange(40)

    joined = standardize_join([a, b], frame)
    assert joined.shape == (50, 10)

    one_block = standardize_join([a], frame)  # weight 1.0 for a single block
    assert abs(one_block[frame].std(axis=0).mean() - 1.0) < 0.2
