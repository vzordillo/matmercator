"""Tests for matmercator.metrics (grid eta^2, k-NN purity, permutations)."""

from __future__ import annotations

import numpy as np

from matmercator.metrics import _bin_index
from matmercator.metrics import grid_eta_squared
from matmercator.metrics import knn_label_purity
from matmercator.metrics import permutation_eta
from matmercator.metrics import permutation_purity


def test_eta_one_when_each_cell_separates_values():
    """Eta^2 = 1 when each occupied cell holds a single distinct value."""
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(grid_eta_squared(coords, values, n_bins=2), 1.0)


def test_eta_zero_when_cell_means_equal_grand_mean():
    """Eta^2 = 0 when every cell mean equals the grand mean."""
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    values = np.array([0.0, 1.0, 0.0, 1.0])  # each cell mean = grand mean = 0.5
    assert np.isclose(grid_eta_squared(coords, values, n_bins=2), 0.0)


def test_eta_nan_without_variance():
    """Eta^2 is NaN when the values have no variance."""
    coords = np.random.default_rng(0).standard_normal((10, 2))
    assert np.isnan(grid_eta_squared(coords, np.full(10, 3.0)))


def test_permutation_eta_detects_signal():
    """A strong cell/value association sits far above the shuffled null."""
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (50, 2))]
    )
    values = np.concatenate([np.zeros(50), np.ones(50)])
    res = permutation_eta(coords, values, n_bins=10, n_perm=100, random_state=0)
    assert set(res) == {
        "observed",
        "null_mean",
        "null_std",
        "z",
        "p_value",
        "n",
    }
    assert res["n"] == 100
    assert res["observed"] > res["null_mean"]
    assert res["z"] > 5
    assert 0 < res["p_value"] <= 1


def test_permutation_eta_observed_matches_grid_eta():
    """The reported observed value equals grid_eta_squared on the input."""
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((60, 2))
    values = rng.standard_normal(60)
    res = permutation_eta(coords, values, n_bins=8, n_perm=20, random_state=3)
    assert np.isclose(res["observed"], grid_eta_squared(coords, values, 8))


def test_eta_null_matches_anova_floor():
    """The eta^2 permutation null mean matches the analytic ANOVA floor.

    Under random labeling the expected between-cell variance fraction is
    ``(n_occupied_cells - 1) / (n - 1)`` -- the same identity RESULTS.md uses to
    show the reported null is correct, not hand-set.
    """
    rng = np.random.default_rng(3)
    n = 1500
    coords = rng.standard_normal((n, 2))
    values = rng.standard_normal(n)  # no real signal
    n_occupied = len(np.unique(_bin_index(coords, 20)))
    analytic = (n_occupied - 1) / (n - 1)
    res = permutation_eta(coords, values, n_bins=20, n_perm=400, random_state=0)
    assert np.isclose(res["null_mean"], analytic, rtol=0.1)
    assert np.isclose(res["observed"], analytic, rtol=0.25)


def test_p_value_floors_at_permutation_resolution():
    """A signal above every permutation gives p = 1 / (n_perm + 1)."""
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.1, (60, 2)), rng.normal(8, 0.1, (60, 2))]
    )
    values = np.concatenate([np.zeros(60), np.ones(60)])
    n_perm = 50
    res = permutation_eta(
        coords, values, n_bins=10, n_perm=n_perm, random_state=0
    )
    assert np.isclose(res["p_value"], 1.0 / (n_perm + 1))


def test_knn_purity_perfect_for_separated_clusters():
    """k-NN purity is 1 when neighbours always share the label."""
    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.05, (20, 2)), rng.normal(9, 0.05, (20, 2))]
    )
    labels = np.array([0] * 20 + [1] * 20)
    assert np.isclose(knn_label_purity(coords, labels, k=3), 1.0)


def test_purity_null_matches_chance_coincidence():
    """The purity permutation null mean matches sum of squared class fractions."""
    rng = np.random.default_rng(4)
    coords = rng.standard_normal((600, 2))
    labels = np.array([0] * 300 + [1] * 180 + [2] * 120)  # 0.5 / 0.3 / 0.2
    res = permutation_purity(coords, labels, k=10, n_perm=300, random_state=0)
    expected = 0.5**2 + 0.3**2 + 0.2**2
    assert np.isclose(res["null_mean"], expected, rtol=0.1)


def test_permutation_purity_schema_and_bounds():
    """permutation_purity returns the documented keys and valid ranges."""
    rng = np.random.default_rng(2)
    coords = np.vstack(
        [rng.normal(0, 0.05, (20, 2)), rng.normal(9, 0.05, (20, 2))]
    )
    labels = np.array([0] * 20 + [1] * 20)
    res = permutation_purity(coords, labels, k=3, n_perm=50, random_state=0)
    assert set(res) == {
        "observed",
        "null_mean",
        "null_std",
        "z",
        "p_value",
        "k",
        "n",
    }
    assert res["k"] == 3 and res["n"] == 40
    assert res["observed"] > res["null_mean"]
    assert 0 < res["p_value"] <= 1


def test_permutation_eta_deterministic():
    """The same seed yields identical permutation results."""
    rng = np.random.default_rng(5)
    coords = rng.standard_normal((40, 2))
    values = rng.standard_normal(40)
    a = permutation_eta(coords, values, n_perm=30, random_state=11)
    b = permutation_eta(coords, values, n_perm=30, random_state=11)
    assert a == b
