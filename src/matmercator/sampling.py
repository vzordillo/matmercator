"""Stratified selection of the frame set.

The frame set must be representative of the corpus so the frozen manifold is not
biased toward common chemistries/symmetries. We allocate at least one sample to
every occupied stratum (so rare space groups are not erased), then distribute
the remaining budget proportionally to stratum size, and sample without
replacement within each stratum.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def stratified_sample_indices(
    df: pd.DataFrame,
    strata: Sequence[str],
    size: int,
    random_state: int = 1234,
) -> np.ndarray:
    """Return row positions into ``df`` for a stratified frame set of ~``size``.

    Args:
        df: The corpus to sample from.
        strata: Column names to stratify by; if empty, falls back to a uniform
            random sample.
        size: Target frame-set size (capped at ``len(df)``).
        random_state: Seed for the random generator.

    Returns:
        A sorted array of integer row positions into ``df``.
    """
    rng = np.random.default_rng(random_state)
    n = len(df)
    size = min(size, n)

    if not strata:
        return np.sort(rng.choice(n, size=size, replace=False))

    # {stratum key: positions ndarray}
    groups = df.groupby(list(strata), sort=False).indices
    keys = list(groups)
    sizes = np.array([len(groups[k]) for k in keys])

    # floor of 1 per occupied stratum, then proportional on the remaining budget
    alloc = np.ones(len(keys), dtype=int)
    remaining = size - alloc.sum()
    if remaining > 0:
        extra = np.floor(remaining * sizes / sizes.sum()).astype(int)
        alloc = alloc + extra
    alloc = np.minimum(alloc, sizes)  # never ask for more than a stratum has

    picks: list = []
    for k, a in zip(keys, alloc, strict=False):
        pos = groups[k]
        picks.extend(rng.choice(pos, size=int(a), replace=False).tolist())
    chosen = np.array(picks, dtype=int)

    # trim any overshoot from the floor+rounding back down to `size`
    if len(chosen) > size:
        chosen = rng.choice(chosen, size=size, replace=False)
    return np.sort(chosen)
