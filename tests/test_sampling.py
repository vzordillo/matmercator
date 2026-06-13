"""Tests for matmercator.sampling.stratified_sample_indices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matmercator.sampling import stratified_sample_indices


def _df(groups):
    """Build a one-column DataFrame whose 'g' column holds the labels."""
    return pd.DataFrame({"g": list(groups)})


def test_returns_sorted_unique_in_range():
    """Indices are sorted, unique, valid row positions of the requested size."""
    df = _df([0] * 50 + [1] * 50)
    idx = stratified_sample_indices(df, ["g"], size=20, random_state=0)
    assert idx.ndim == 1
    assert np.all(np.diff(idx) > 0)  # strictly increasing => sorted & unique
    assert idx.min() >= 0 and idx.max() < len(df)
    assert len(idx) == 20


def test_rare_stratum_is_represented():
    """Every occupied stratum receives at least one sample."""
    df = _df(["rare"] + ["common"] * 199)
    idx = stratified_sample_indices(df, ["g"], size=10, random_state=0)
    assert 0 in idx  # the single 'rare' row sits at position 0


def test_proportional_allocation():
    """Beyond the floor of one, the budget is allocated proportionally."""
    df = _df([0] * 100 + [1] * 900)
    idx = stratified_sample_indices(df, ["g"], size=100, random_state=0)
    g = df["g"].to_numpy()[idx]
    n0, n1 = int((g == 0).sum()), int((g == 1).sum())
    assert abs(n0 - 10) <= 2  # ~10% of the budget to the 10% stratum
    assert abs(n1 - 90) <= 2  # ~90% to the 90% stratum
    assert n0 + n1 == len(idx)


def test_size_capped_at_population():
    """Requesting more than the population returns the whole population."""
    df = _df([0, 1, 2, 3, 4])
    idx = stratified_sample_indices(df, ["g"], size=99, random_state=0)
    assert list(idx) == [0, 1, 2, 3, 4]


def test_empty_strata_uniform_sample():
    """With no strata, an exact-size uniform sample is drawn."""
    df = _df(range(100))
    idx = stratified_sample_indices(df, [], size=15, random_state=0)
    assert len(idx) == 15
    assert len(set(idx.tolist())) == 15


def test_deterministic_given_seed():
    """The same seed yields identical indices."""
    df = _df([0] * 40 + [1] * 60)
    a = stratified_sample_indices(df, ["g"], size=25, random_state=7)
    b = stratified_sample_indices(df, ["g"], size=25, random_state=7)
    assert np.array_equal(a, b)
