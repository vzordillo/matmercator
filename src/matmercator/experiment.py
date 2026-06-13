"""Descriptor comparison: is SCM mostly composition? does GTM beat PCA?

For each descriptor (SCM, composition, their union) the same GTM is fit on the
frame set and three held-out Q^2 flavors are reported per property:

  * ``q2_gtm_resp`` — the GTM's own responsibility-based predictor (q2_cv);
  * ``q2_gtm_2d``   — k-NN regression on the GTM 2-D embedding;
  * ``q2_pca_2d``   — k-NN regression on a PCA 2-D embedding (the linear baseline).

Comparing ``q2_gtm_2d`` vs ``q2_pca_2d`` isolates "is the GTM embedding better
than a linear one"; ``q2_gtm_resp`` is the GTM's full predictive power. The
``n_sites`` confound (how well map position predicts cell size) flags a map that
is partly a size map. All arms share one frame-fit, block-balanced
standardization (each block z-scored then scaled by 1/sqrt(d_block)) so the
comparison is fair; GTM's own standardizer is therefore disabled here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from matmercator.cartography import GTMCartographer
from matmercator.config import PipelineConfig
from matmercator.featurize_cache import cache_dir_for
from matmercator.sampling import stratified_sample_indices
from matmercator.selection import q2_cv

log = logging.getLogger("matmercator")


def standardize_join(blocks, frame_idx, weights=None) -> np.ndarray:
    """Per-block frame-fit z-score, weight, and horizontally concatenate.

    Args:
        blocks: List of ``(n, d_b)`` descriptor blocks (row-aligned).
        frame_idx: Row indices used to fit each block's scaler (the frame set).
        weights: Per-block multipliers. Default: 1 for a single block, else
            ``1/sqrt(d_b)`` so each block contributes comparable total variance.

    Returns:
        The standardized, weighted, concatenated ``(n, sum d_b)`` matrix.
    """
    if weights is None:
        weights = (
            [1.0]
            if len(blocks) == 1
            else [1.0 / np.sqrt(b.shape[1]) for b in blocks]
        )
    parts = []
    for block, w in zip(blocks, weights, strict=True):
        scaler = StandardScaler().fit(block[frame_idx])
        parts.append(scaler.transform(block) * w)
    return np.hstack(parts)


def q2_knn(coords, y, n_neighbors=15, n_folds=5, seed=1234) -> float:
    """Cross-validated Q^2 of k-NN regression of ``y`` on a 2-D embedding."""
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    coords, y = coords[finite], y[finite]
    if len(y) < n_folds or np.ptp(y) == 0:
        return float("nan")
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    model = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(y) - 1))
    y_pred = cross_val_predict(model, coords, y, cv=cv)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_descriptor(cfg, name, blocks, df, frame_idx, weights=None) -> dict:
    """Fit GTM on one descriptor and score the three Q^2 flavors + confound."""
    cfg = cfg.resolved()
    Xj = standardize_join(blocks, frame_idx, weights)
    carto = GTMCartographer(
        k=cfg.gtm_k,
        m=cfg.gtm_m,
        s=cfg.gtm_s,
        regul=cfg.gtm_regul,
        niter=cfg.gtm_niter,
        random_state=cfg.random_state,
        standardize=False,  # already standardized + block-weighted above
    ).fit(Xj[frame_idx])
    R = carto.responsibilities(Xj)
    coords = R @ carto.node_coords
    pca_coords = (
        PCA(n_components=2, random_state=cfg.random_state)
        .fit(Xj[frame_idx])
        .transform(Xj)
    )

    props = list(cfg.color_properties)
    per_prop = {}
    for p in props:
        y = df[p].to_numpy()
        per_prop[p] = {
            "q2_gtm_resp": q2_cv(R, y, seed=cfg.random_state),
            "q2_gtm_2d": q2_knn(coords, y, seed=cfg.random_state),
            "q2_pca_2d": q2_knn(pca_coords, y, seed=cfg.random_state),
        }
    mean_resp = float(np.nanmean([per_prop[p]["q2_gtm_resp"] for p in props]))
    n_sites_q2 = (
        q2_knn(coords, df["n_sites"].to_numpy(), seed=cfg.random_state)
        if "n_sites" in df
        else float("nan")
    )
    log.info(
        "[%-15s] dims=%3d  mean GTM-resp Q2=%+.3f  n_sites Q2(2d)=%+.3f",
        name,
        Xj.shape[1],
        mean_resp,
        n_sites_q2,
    )
    return {
        "descriptor": name,
        "n_features": int(Xj.shape[1]),
        "mean_q2_gtm_resp": mean_resp,
        "n_sites_q2_gtm_2d": n_sites_q2,
        "properties": per_prop,
    }


def _format_md(report: dict) -> str:
    """Render the experiment report as a Markdown summary."""
    lines = [
        "# Descriptor experiment — MP-20",
        "",
        f"{report['n_structures']} structures, frame set "
        f"{report['n_frame_set']}, GTM k={report['gtm']['k']} "
        f"niter={report['gtm']['niter']}. "
        "Q2 is held-out (5-fold); higher = more predictive.",
        "",
        "## Mean GTM-responsibility Q2 (across properties) + size confound",
        "",
        "| descriptor | dims | mean Q2 | n_sites Q2 |",
        "|---|---|---|---|",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['descriptor']} | {arm['n_features']} | "
            f"{arm['mean_q2_gtm_resp']:+.3f} | "
            f"{arm['n_sites_q2_gtm_2d']:+.3f} |"
        )
    lines += ["", "## Per-property Q2 (GTM-resp / GTM-2D / PCA-2D)", ""]
    props = report["arms"][0]["properties"].keys()
    header = "| descriptor | " + " | ".join(props) + " |"
    lines += [header, "|" + "---|" * (len(list(props)) + 1)]
    for arm in report["arms"]:
        cells = []
        for p in arm["properties"]:
            d = arm["properties"][p]
            cells.append(
                f"{d['q2_gtm_resp']:+.2f}/{d['q2_gtm_2d']:+.2f}/"
                f"{d['q2_pca_2d']:+.2f}"
            )
        lines.append(f"| {arm['descriptor']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run_experiment(cfg: PipelineConfig) -> dict:
    """Run the SCM vs composition vs union comparison from the feature cache.

    Requires the SCM cache (``X_{split}.npz``) and the composition cache
    (``Xcomp_{split}.npz``; see :func:`matmercator.composition.cache_composition`)
    plus the per-split metadata. Writes ``experiment_report.json`` and
    ``experiment.md`` to ``cfg.output_dir``.

    Args:
        cfg: Run configuration.

    Returns:
        The experiment report dict.
    """
    cfg = cfg.resolved()
    cache = cache_dir_for(cfg)
    metas, scm, comp = [], [], []
    for split in cfg.splits:
        metas.append(pd.read_parquet(cache / f"meta_{split}.parquet"))
        scm.append(np.load(cache / f"X_{split}.npz")["X"])
        comp.append(np.load(cache / f"Xcomp_{split}.npz")["X"])
    df = pd.concat(metas, ignore_index=True)
    x_scm = np.vstack(scm).astype(float)
    x_scm = x_scm[:, np.abs(x_scm).sum(axis=0) > 0]
    x_comp = np.vstack(comp).astype(float)

    pool = df.index[df["split"] == cfg.frame_split].to_numpy()
    frame_df = df.loc[pool].reset_index(drop=True)
    rel = stratified_sample_indices(
        frame_df, cfg.frame_strata, cfg.frame_set_size, cfg.random_state
    )
    frame_idx = pool[rel]

    arms = [
        ("scm", [x_scm], None),
        ("composition", [x_comp], None),
        ("scm+composition", [x_scm, x_comp], None),
    ]
    results = [
        run_descriptor(cfg, name, blocks, df, frame_idx, weights)
        for name, blocks, weights in arms
    ]
    report = {
        "n_structures": int(len(df)),
        "n_frame_set": int(len(frame_idx)),
        "gtm": {
            "k": cfg.gtm_k,
            "m": cfg.gtm_m,
            "s": cfg.gtm_s,
            "regul": cfg.gtm_regul,
            "niter": cfg.gtm_niter,
        },
        "arms": results,
    }
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_report.json").write_text(json.dumps(report, indent=2))
    (out / "experiment.md").write_text(_format_md(report))
    return report
