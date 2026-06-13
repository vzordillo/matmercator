"""Tests for matmercator.cartography.GTMCartographer."""

from __future__ import annotations

import numpy as np
import pytest

from matmercator.cartography import GTMCartographer


def _toy_data(seed=0, n=60, d=5):
    """Return a small random descriptor matrix for GTM fitting."""
    return np.random.default_rng(seed).standard_normal((n, d))


def test_methods_raise_before_fit():
    """responsibilities/project raise RuntimeError before fit."""
    carto = GTMCartographer(k=4, m=2, niter=10)
    with pytest.raises(RuntimeError):
        carto.responsibilities(_toy_data())
    with pytest.raises(RuntimeError):
        carto.project(_toy_data())


def test_responsibilities_are_valid_posteriors():
    """Responsibilities are (n, k^2), non-negative, and sum to 1 per row."""
    X = _toy_data()
    carto = GTMCartographer(k=4, m=2, niter=20, random_state=0).fit(X[:30])
    R = carto.responsibilities(X)
    assert R.shape == (len(X), 16)
    assert R.min() >= -1e-9
    np.testing.assert_allclose(R.sum(axis=1), np.ones(len(X)), atol=1e-5)
    assert carto.n_nodes == 16
    assert carto.node_coords.shape == (16, 2)


def test_project_equals_responsibility_weighted_nodes():
    """project(X) equals responsibilities(X) @ node_coords (the GTM identity)."""
    X = _toy_data()
    carto = GTMCartographer(k=4, m=2, niter=20, random_state=0).fit(X[:30])
    R = carto.responsibilities(X)
    np.testing.assert_allclose(
        carto.project(X), R @ carto.node_coords, atol=1e-9
    )


def test_projection_is_deterministic():
    """Two fits with the same seed give identical projections."""
    X = _toy_data()
    a = GTMCartographer(k=4, m=2, niter=20, random_state=0).fit_project(
        X[:30], X
    )
    b = GTMCartographer(k=4, m=2, niter=20, random_state=0).fit_project(
        X[:30], X
    )
    np.testing.assert_allclose(a, b)


def test_standardize_false_runs():
    """The no-standardize path fits and projects without error."""
    X = _toy_data()
    carto = GTMCartographer(k=4, m=2, niter=20, standardize=False).fit(X[:30])
    assert carto.project(X).shape == (len(X), 2)


def test_recovers_planted_clusters():
    """Two well-separated feature clusters land far apart on the 2-D map."""
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(-10.0, 0.5, (40, 5))
    cluster_b = rng.normal(10.0, 0.5, (40, 5))
    X = np.vstack([cluster_a, cluster_b])
    coords = GTMCartographer(k=6, m=2, niter=40, random_state=0).fit_project(
        X, X
    )
    ca, cb = coords[:40].mean(axis=0), coords[40:].mean(axis=0)
    between = np.linalg.norm(ca - cb)
    within = np.concatenate(
        [
            np.linalg.norm(coords[:40] - ca, axis=1),
            np.linalg.norm(coords[40:] - cb, axis=1),
        ]
    ).mean()
    assert between > 2.0 * within
