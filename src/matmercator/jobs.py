"""Cache-consuming orchestration: build the map and landscapes from the cache.

These functions are the staged counterparts to :func:`matmercator.pipeline.run`:
they read the feature cache written by :mod:`matmercator.featurize_cache`
(validating the descriptor identity) and build the map, the node-based
landscapes, or fit a cartographer for downstream rendering. The cache directory
sits beside ``output_dir`` (``<output_dir>/../cache``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from matmercator.cache import check_descriptor
from matmercator.cache import descriptor_spec
from matmercator.cartography import GTMCartographer
from matmercator.config import PipelineConfig
from matmercator.featurize_cache import cache_dir_for
from matmercator.pipeline import run_from_features
from matmercator.sampling import stratified_sample_indices

log = logging.getLogger("matmercator")


def load_cache(cfg: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and assemble the staged feature cache for ``cfg``.

    Validates that the cache's descriptor identity matches the config, then
    concatenates the per-split shards and drops globally all-zero pad columns.

    Args:
        cfg: Run configuration.

    Returns:
        ``(df, X)`` — the metadata table and the row-aligned descriptor matrix.
    """
    cfg = cfg.resolved()
    cache = cache_dir_for(cfg)
    check_descriptor(
        cache, descriptor_spec(cfg.diag_elems, cfg.sort_eigenvalues)
    )
    metas, mats = [], []
    for split in cfg.splits:
        metas.append(pd.read_parquet(cache / f"meta_{split}.parquet"))
        mats.append(np.load(cache / f"X_{split}.npz")["X"])
    df = pd.concat(metas, ignore_index=True)
    X = np.vstack(mats).astype(float)
    X = X[:, np.abs(X).sum(axis=0) > 0]
    log.info("loaded cache: %d structures, %d features", len(df), X.shape[1])
    return df, X


def map_from_cache(cfg: PipelineConfig) -> dict:
    """Build the map from the cache (staged equivalent of ``pipeline.run``)."""
    df, X = load_cache(cfg)
    return run_from_features(cfg, df, X)


def fit_cartographer_from_cache(
    cfg: PipelineConfig,
) -> tuple[pd.DataFrame, np.ndarray, GTMCartographer, np.ndarray]:
    """Load the cache and fit a frozen cartographer on the stratified frame set.

    Args:
        cfg: Run configuration.

    Returns:
        ``(df, X, carto, R)`` — metadata, descriptors, the fitted cartographer,
        and the ``(n, K)`` responsibility matrix for the full corpus.
    """
    cfg = cfg.resolved()
    df, X = load_cache(cfg)
    pool = df.index[df["split"] == cfg.frame_split].to_numpy()
    frame_df = df.loc[pool].reset_index(drop=True)
    rel = stratified_sample_indices(
        frame_df, cfg.frame_strata, cfg.frame_set_size, cfg.random_state
    )
    carto = GTMCartographer(
        k=cfg.gtm_k,
        m=cfg.gtm_m,
        s=cfg.gtm_s,
        regul=cfg.gtm_regul,
        niter=cfg.gtm_niter,
        random_state=cfg.random_state,
        standardize=cfg.standardize,
    ).fit(X[pool[rel]])
    R = carto.responsibilities(X)
    return df, X, carto, R


# Vivid, perceptually-uniform colormaps for the continuous landscapes
# (no pale midpoint like coolwarm).
_LANDSCAPE_CMAP = {
    "band_gap": "viridis",
    "formation_energy_per_atom": "plasma",
    "e_above_hull": "magma",
}


def landscapes_from_cache(cfg: PipelineConfig) -> dict[str, str]:
    """Render the node-based landscapes from the cache.

    Writes per-property density/coherence/applicability panels, a two-class
    metal/non-metal landscape, and a crystal-system winning-class map to
    ``<output_dir>/landscapes/``.

    Args:
        cfg: Run configuration.

    Returns:
        A mapping of landscape name to written PNG path.
    """
    from matmercator import landscape as L
    from matmercator.plots import _PROP_META
    from matmercator.plots import CRYSTAL_SYSTEMS
    from matmercator.plots import crystal_system

    cfg = cfg.resolved()
    out = Path(cfg.output_dir) / "landscapes"
    out.mkdir(parents=True, exist_ok=True)

    df, _, carto, R = fit_cartographer_from_cache(cfg)
    nc = carto.node_coords
    paths: dict[str, str] = {}

    for prop in cfg.color_properties:
        label, _ = _PROP_META.get(prop, (prop, "viridis"))
        cmap = _LANDSCAPE_CMAP.get(prop, "viridis")
        paths[prop] = L.property_landscape_panel(
            R, nc, df[prop].to_numpy(), prop, label, cmap, out
        )

    metal = np.where(df["band_gap"].to_numpy() < 0.01, 1.0, 2.0)
    paths["metal_nonmetal"] = L.two_class_landscape(
        R,
        nc,
        metal,
        out,
        labels=("metal", "non-metal"),
        name="metal_nonmetal",
    )

    csys = df["spacegroup.number"].map(crystal_system).to_numpy()
    paths["crystal_system"] = L.winning_class_landscape(
        R, nc, csys, CRYSTAL_SYSTEMS, out, name="crystal_system"
    )
    log.info("wrote %d landscapes to %s", len(paths), out)
    return paths
