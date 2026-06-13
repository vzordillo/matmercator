"""Quantitative map-quality checks.

A map that *looks* organized is not necessarily meaningful. These metrics ask:
does position on the GTM landscape carry information about a property, beyond
what you'd get by chance? Each metric is paired with a label-permutation null,
so the reported effect is measured against an explicit baseline rather than
asserted.

Two metrics:
  * ``grid_eta_squared`` (continuous properties): bin the 2-D embedding into a
    G x G grid and compute the fraction of property variance that lies BETWEEN
    cells (a one-way ANOVA eta^2). eta^2 = 1 means cell membership fully
    determines the property; eta^2 ~ 0 means the map says nothing about it.
  * ``knn_label_purity`` (categorical labels, e.g. crystal system): the mean
    fraction of each point's k nearest map-neighbours that share its label.

Permutation tests shuffle the property/label across points and recompute, giving
a null distribution, a z-score, and an empirical one-sided p-value.
"""

from __future__ import annotations

import numpy as np


def _bin_index(coords: np.ndarray, n_bins: int) -> np.ndarray:
    """Map ``(n, 2)`` coords to a flat cell id in ``[0, n_bins**2)``.

    Args:
        coords: ``(n, 2)`` map coordinates.
        n_bins: Number of bins per axis.

    Returns:
        An ``(n,)`` integer array of flat cell ids.
    """
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    frac = (coords - lo) / span
    ij = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    return ij[:, 0] * n_bins + ij[:, 1]


def _eta_from_cell(
    cell: np.ndarray, values: np.ndarray, minlength: int
) -> float:
    """eta^2 given precomputed cell ids -- the hot path for permutations.

    Args:
        cell: ``(n,)`` flat cell ids.
        values: ``(n,)`` property values aligned to ``cell``.
        minlength: ``n_bins**2``, so empty trailing cells are counted.

    Returns:
        The between-cell fraction of variance, or NaN if total variance is 0.
    """
    grand = values.mean()
    ss_tot = np.sum((values - grand) ** 2)
    if ss_tot <= 0:
        return float("nan")
    counts = np.bincount(cell, minlength=minlength)
    sums = np.bincount(cell, weights=values, minlength=minlength)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    ss_between = np.sum(counts * (means - grand) ** 2)  # empty cells add 0
    return float(ss_between / ss_tot)


def grid_eta_squared(coords, values, n_bins: int = 20) -> float:
    """Between-cell fraction of variance (eta^2) over a G x G grid.

    Args:
        coords: ``(n, 2)`` map coordinates.
        values: ``(n,)`` property values; non-finite entries are dropped.
        n_bins: Number of bins per axis.

    Returns:
        The eta^2 statistic, or NaN if fewer than two finite values remain.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    coords, values = coords[mask], values[mask]
    if len(values) < 2:
        return float("nan")
    cell = _bin_index(coords, n_bins)
    return _eta_from_cell(cell, values, n_bins * n_bins)


def permutation_eta(
    coords, values, n_bins: int = 20, n_perm: int = 200, random_state: int = 0
) -> dict[str, float]:
    """eta^2 against a label-shuffled null distribution.

    Args:
        coords: ``(n, 2)`` map coordinates.
        values: ``(n,)`` property values; non-finite entries are dropped.
        n_bins: Number of bins per axis.
        n_perm: Number of label permutations forming the null.
        random_state: Seed for the permutations.

    Returns:
        A dict with the observed eta^2, the null mean/std, a z-score, an
        empirical one-sided p-value, and the sample size ``n``.
    """
    rng = np.random.default_rng(random_state)
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    coords, values = coords[mask], values[mask]

    # Binning is fixed under label permutation -> compute cell ids once.
    cell = _bin_index(coords, n_bins)
    minlen = n_bins * n_bins
    obs = _eta_from_cell(cell, values, minlen)
    null = np.array(
        [
            _eta_from_cell(cell, rng.permutation(values), minlen)
            for _ in range(n_perm)
        ]
    )
    null_mean, null_std = float(null.mean()), float(null.std(ddof=1))
    z = (obs - null_mean) / null_std if null_std > 0 else float("nan")
    p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)
    return dict(
        observed=obs,
        null_mean=null_mean,
        null_std=null_std,
        z=float(z),
        p_value=float(p),
        n=int(len(values)),
    )


def _knn_index(coords: np.ndarray, k: int) -> np.ndarray:
    """Find each point's k nearest neighbours (self excluded).

    Args:
        coords: ``(n, 2)`` map coordinates.
        k: Number of neighbours.

    Returns:
        An ``(n, k)`` array of neighbour row indices.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]  # drop self


def knn_label_purity(coords, labels, k: int = 15) -> float:
    """Mean fraction of each point's k map-neighbours sharing its label.

    Args:
        coords: ``(n, 2)`` map coordinates.
        labels: ``(n,)`` categorical labels.
        k: Number of neighbours (capped at ``n - 1``).

    Returns:
        The mean k-NN label purity in ``[0, 1]``.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    k = min(k, len(coords) - 1)
    idx = _knn_index(coords, k)
    return float((labels[idx] == labels[:, None]).mean())


def permutation_purity(
    coords, labels, k: int = 15, n_perm: int = 200, random_state: int = 0
) -> dict[str, float]:
    """k-NN label purity against a label-shuffled null distribution.

    Args:
        coords: ``(n, 2)`` map coordinates.
        labels: ``(n,)`` categorical labels.
        k: Number of neighbours (capped at ``n - 1``).
        n_perm: Number of label permutations forming the null.
        random_state: Seed for the permutations.

    Returns:
        A dict with the observed purity, the null mean/std, a z-score, an
        empirical one-sided p-value, the realized ``k``, and ``n``.
    """
    rng = np.random.default_rng(random_state)
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels)
    k = min(k, len(coords) - 1)

    # The neighbour graph is fixed under label permutation -> build it once.
    idx = _knn_index(coords, k)
    obs = float((labels[idx] == labels[:, None]).mean())
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(labels)
        null[i] = (perm[idx] == perm[:, None]).mean()
    null_mean, null_std = float(null.mean()), float(null.std(ddof=1))
    z = (obs - null_mean) / null_std if null_std > 0 else float("nan")
    p = (1 + int(np.sum(null >= obs))) / (n_perm + 1)
    return dict(
        observed=obs,
        null_mean=null_mean,
        null_std=null_std,
        z=float(z),
        p_value=float(p),
        k=int(k),
        n=int(len(labels)),
    )
