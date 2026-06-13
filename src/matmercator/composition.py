"""Composition (Magpie) descriptor — a structure-free baseline.

matminer's ``ElementProperty`` "magpie" preset: summary statistics over the
elemental properties of a formula (132 features). It needs only the composition,
so it is a cheap baseline for asking how much of a map is driven by chemistry
alone versus geometry. Built from ``pretty_formula`` (no CIF parsing required),
which is why it can be computed straight from the cached metadata.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

from matmercator.config import PipelineConfig
from matmercator.featurize_cache import cache_dir_for

log = logging.getLogger("matmercator")


class CompositionFeaturizer:
    """Fixed-length Magpie composition descriptor from chemical formulas."""

    def __init__(self, preset: str = "magpie", n_jobs: int = 1):
        """Initialize the matminer ElementProperty featurizer.

        Args:
            preset: ElementProperty preset name (default ``"magpie"``).
            n_jobs: Worker processes for ``featurize_many``. Magpie features are
                per-composition and order-preserving, so n_jobs > 1 stays
                deterministic (useful for caching large sets).
        """
        self._ep = ElementProperty.from_preset(preset)
        self._ep.set_n_jobs(n_jobs)
        self.n_features_ = len(self._ep.feature_labels())

    def transform(self, formulas) -> np.ndarray:
        """Featurize an iterable of formula strings -> ``(n, 132)``.

        Any NaN cells (rare; some Magpie stats are undefined for exotic
        elements) are filled with the column mean and the count is logged --
        explicit, never silent.

        Args:
            formulas: Iterable of chemical-formula strings.

        Returns:
            The ``(n, n_features_)`` float descriptor matrix.
        """
        comps = [Composition(f) for f in formulas]
        X = np.asarray(self._ep.featurize_many(comps, pbar=False), dtype=float)
        nan = np.isnan(X)
        if nan.any():
            col_mean = np.nanmean(X, axis=0)
            X[nan] = np.take(col_mean, np.where(nan)[1])
            log.warning(
                "composition: filled %d NaN cells with column means",
                int(nan.sum()),
            )
        return X


def cache_composition(
    cfg: PipelineConfig, out_dir=None, procs: int = 1
) -> None:
    """Compute and cache Magpie composition features for each split.

    Reads the per-split ``meta_{split}.parquet`` (for ``pretty_formula``) from
    the cache and writes ``Xcomp_{split}.npz`` beside it. Fast — no CIF parsing.

    Defaults to serial (``procs=1``): Magpie is ~1 ms/structure, and matminer's
    multiprocessing pickles every ``Composition`` to workers, which is slower in
    practice and can stall under macOS ``spawn`` (especially when launched from a
    ``python -c`` one-liner with no importable ``__main__``). Raise ``procs``
    only from inside a script guarded by ``if __name__ == "__main__":``.

    Args:
        cfg: Run configuration (supplies splits and the cache location).
        out_dir: Cache directory (defaults to :func:`cache_dir_for`).
        procs: Worker processes (default 1 = serial; see note above).
    """
    cfg = cfg.resolved()
    out = out_dir if out_dir is not None else cache_dir_for(cfg)
    feat = CompositionFeaturizer(n_jobs=procs)
    for split in cfg.splits:
        meta = pd.read_parquet(out / f"meta_{split}.parquet")
        X = feat.transform(meta["pretty_formula"].tolist())
        np.savez_compressed(out / f"Xcomp_{split}.npz", X=X.astype(np.float32))
        log.info("composition cache %s: %s", split, X.shape)
