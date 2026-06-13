"""Q2-driven GTM map selection — Horvath's map-quality criterion.

A GTM map doubles as a property predictor: a structure's responsibilities place
it on the map, and the prediction is the responsibility-weighted average of the
per-node mean property,

    P_hat(n) = sum_k R_kn * Pbar_k .

``q2_cv`` scores how well that predicts a held-out property via k-fold
cross-validation (the cross-validated R^2, i.e. Q^2), computing the node means
``Pbar_k`` from training-fold structures only so no label leaks into the
prediction. The manifold itself is unsupervised (fit on descriptors, never on
the property), so fitting it on the frame set introduces no label leakage.

``select_gtm`` sweeps GTM hyperparameters (k, m, s, regul) and ranks them by the
mean Q^2 across properties — turning "which map looks nicer" into an objective,
out-of-sample comparison that also penalizes over-fine grids (unlike the
in-sample grid-eta^2 metric). Q^2 for an unsupervised SCM map is expected to be
low/modest; use it for *relative* comparison, not as an absolute predictor claim.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field

import numpy as np
from sklearn.model_selection import KFold

from matmercator.cartography import GTMCartographer
from matmercator.config import PipelineConfig
from matmercator.sampling import stratified_sample_indices

log = logging.getLogger("matmercator")


def q2_cv(
    R: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 1234,
) -> float:
    """Cross-validated Q^2 of the global GTM prediction for one property.

    For each fold the per-node means ``Pbar_k`` are computed from the training
    rows only, then test rows are predicted by ``R_test @ Pbar`` — no label
    leakage. Non-finite property values are dropped.

    Args:
        R: ``(n, K)`` responsibility matrix.
        y: ``(n,)`` property values aligned to ``R``'s rows.
        n_folds: Number of CV folds.
        seed: Shuffle seed for the folds.

    Returns:
        Q^2 = 1 - SS_res/SS_tot over the held-out predictions (1 perfect, 0 = no
        better than the global mean, < 0 worse). NaN if the property has no
        variance or there are too few finite values.
    """
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    R, y = R[finite], y[finite]
    if len(y) < n_folds or np.ptp(y) == 0:
        return float("nan")

    y_pred = np.empty(len(y))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train, test in kf.split(R):
        denom = np.maximum(R[train].sum(axis=0), 1e-12)  # cumulated resp.
        node_mean = (R[train].T @ y[train]) / denom
        y_pred[test] = R[test] @ node_mean

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass
class GTMSearchSpace:
    """Grid of GTM hyperparameters to sweep (Cartesian product)."""

    k: Sequence[int] = (12, 16, 20)
    m: Sequence[int] = (3, 4)
    s: Sequence[float] = (0.2, 0.3)
    regul: Sequence[float] = field(default_factory=lambda: (0.1,))


def select_gtm(
    cfg: PipelineConfig,
    df,
    X: np.ndarray,
    *,
    space: GTMSearchSpace | None = None,
    n_folds: int = 5,
) -> dict:
    """Rank GTM hyperparameters by mean cross-validated Q^2 across properties.

    The descriptor matrix and the stratified frame set are computed once; each
    candidate refits only the manifold (on the frame set) and scores the
    coloring by :func:`q2_cv`.

    Args:
        cfg: Run configuration (frame-set, properties, seed, niter, standardize).
        df: Metadata table row-aligned to ``X`` (needs ``split`` and the
            configured ``color_properties``).
        X: ``(n, d)`` descriptor matrix.
        space: Hyperparameter grid (defaults to :class:`GTMSearchSpace`).
        n_folds: CV folds for ``q2_cv``.

    Returns:
        A dict with the best row, the full scored ``table`` (one row per
        candidate with per-property and mean Q^2), and run metadata.
    """
    cfg = cfg.resolved()
    space = space or GTMSearchSpace()
    props = list(cfg.color_properties)

    pool = df.index[df["split"] == cfg.frame_split].to_numpy()
    frame_df = df.loc[pool].reset_index(drop=True)
    rel = stratified_sample_indices(
        frame_df, cfg.frame_strata, cfg.frame_set_size, cfg.random_state
    )
    frame_idx = pool[rel]

    table: list[dict] = []
    best: dict | None = None
    for k in space.k:
        for m in space.m:
            for s in space.s:
                for regul in space.regul:
                    carto = GTMCartographer(
                        k=k,
                        m=m,
                        s=s,
                        regul=regul,
                        niter=cfg.gtm_niter,
                        random_state=cfg.random_state,
                        standardize=cfg.standardize,
                    ).fit(X[frame_idx])
                    R = carto.responsibilities(X)
                    q2 = {
                        p: q2_cv(R, df[p].to_numpy(), n_folds, cfg.random_state)
                        for p in props
                    }
                    mean_q2 = float(np.nanmean(list(q2.values())))
                    row = {
                        "k": k,
                        "m": m,
                        "s": s,
                        "regul": regul,
                        "q2": q2,
                        "mean_q2": mean_q2,
                    }
                    table.append(row)
                    log.info(
                        "k=%2d m=%d s=%.2f regul=%.3f -> mean Q2=%+.3f",
                        k,
                        m,
                        s,
                        regul,
                        mean_q2,
                    )
                    if best is None or mean_q2 > best["mean_q2"]:
                        best = row

    return {
        "best": best,
        "table": table,
        "n_folds": n_folds,
        "properties": props,
        "n_structures": int(len(df)),
        "n_frame_set": int(len(frame_idx)),
    }
