"""End-to-end MP-20 cartography pipeline.

    load  ->  featurize  ->  fit GTM on a stratified frame set
          ->  project the full set  ->  color-code  ->  validate

Everything that defines the frozen map (the StandardScaler and the GTM) is fit
on the frame set only; the full corpus is projected onto it. Outputs (projected
coordinates as Parquet, PNG maps, and a metrics/JSON report) are written to
``config.output_dir``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from matmercator.cartography import GTMCartographer
from matmercator.config import PipelineConfig
from matmercator.data import load_dataset
from matmercator.featurize import SCMFeaturizer
from matmercator.metrics import permutation_eta
from matmercator.metrics import permutation_purity
from matmercator.plots import crystal_system
from matmercator.plots import plot_property_maps
from matmercator.provenance import collect_provenance
from matmercator.sampling import stratified_sample_indices

log = logging.getLogger("matmercator")


def build_features(cfg: PipelineConfig):
    """Steps 1-2: load + featurize.

    Args:
        cfg: The run configuration.

    Returns:
        A tuple ``(df, X, timings)`` of the loaded DataFrame, the ``(n, d)``
        descriptor matrix, and a dict of per-stage wall times.
    """
    timings: dict[str, float] = {}
    t = time.time()
    df = load_dataset(
        cfg.data_root, cfg.dataset, cfg.splits, cfg.max_structures
    )
    timings["load_parse_s"] = time.time() - t
    log.info(
        "loaded %d %s structures in %.1fs",
        len(df),
        cfg.dataset,
        timings["load_parse_s"],
    )

    t = time.time()
    feat = SCMFeaturizer(
        diag_elems=cfg.diag_elems, sort_eigenvalues=cfg.sort_eigenvalues
    )
    X = feat.fit_transform(df["structure"].tolist())
    timings["featurize_s"] = time.time() - t
    log.info("featurized -> %s in %.1fs", X.shape, timings["featurize_s"])
    return df, X, timings


def run(config: PipelineConfig) -> dict:
    """Canonical single-process entry point: load -> featurize -> map.

    Args:
        config: The run configuration.

    Returns:
        The metrics/summary report dict (also written to ``report.json``).
    """
    cfg = config.resolved()
    df, X, timings = build_features(cfg)
    return run_from_features(cfg, df, X, timings=timings)


def run_from_features(
    config: PipelineConfig,
    df: pd.DataFrame,
    X: np.ndarray,
    timings: dict | None = None,
) -> dict:
    """Steps 3-7: frame-set GTM fit -> project -> color-code -> validate.

    This is the shared map-building path; both the canonical in-memory run and
    the staged (cached-feature) run funnel through here.

    Writes ``config.json``, ``gtm_coords.parquet``, the ``map_*.png`` figures
    and ``report.json`` to ``config.output_dir``. The ``gtm_coords.parquet``
    table has one row per structure with columns: ``material_id``,
    ``pretty_formula``, ``split``, ``n_sites``, ``spacegroup.number``, the
    configured ``color_properties``, ``gtm_x``, ``gtm_y``, ``crystal_system``
    and ``in_frame_set`` (True for the frame-set rows).

    Args:
        config: The run configuration.
        df: Row-aligned to ``X``, carrying the columns ``split``,
            ``spacegroup.number``, ``material_id``, ``pretty_formula``,
            ``n_sites`` and the configured ``color_properties``.
        X: ``(n, d)`` descriptor matrix.
        timings: Optional per-stage wall times to extend (e.g. from
            ``build_features``).

    Returns:
        The metrics/summary report dict.
    """
    cfg = config.resolved()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")
    timings = dict(timings or {})
    X = np.asarray(X, dtype=float)

    # 3. frame set ------------------------------------------------------------
    frame_pool = df.index[df["split"] == cfg.frame_split].to_numpy()
    frame_df = df.loc[frame_pool].reset_index(drop=True)
    rel = stratified_sample_indices(
        frame_df, cfg.frame_strata, cfg.frame_set_size, cfg.random_state
    )
    frame_idx = frame_pool[rel]  # positions into the full df / X

    # 4. fit GTM on frame set, project everything -----------------------------
    t = time.time()
    carto = GTMCartographer(
        k=cfg.gtm_k,
        m=cfg.gtm_m,
        s=cfg.gtm_s,
        regul=cfg.gtm_regul,
        niter=cfg.gtm_niter,
        random_state=cfg.random_state,
        standardize=cfg.standardize,
    )
    carto.fit(X[frame_idx])
    timings["gtm_fit_s"] = time.time() - t
    log.info(
        "fit GTM (k=%d) on frame set of %d in %.1fs",
        cfg.gtm_k,
        len(frame_idx),
        timings["gtm_fit_s"],
    )

    t = time.time()
    coords = carto.project(X)
    timings["project_s"] = time.time() - t
    log.info("projected %d structures in %.2fs", len(df), timings["project_s"])

    # 5. persist coordinates --------------------------------------------------
    coord_df = df[
        [
            "material_id",
            "pretty_formula",
            "split",
            "n_sites",
            "spacegroup.number",
            *cfg.color_properties,
        ]
    ].copy()
    coord_df["gtm_x"] = coords[:, 0]
    coord_df["gtm_y"] = coords[:, 1]
    coord_df["crystal_system"] = coord_df["spacegroup.number"].map(
        crystal_system
    )
    coord_df["in_frame_set"] = False
    coord_df.loc[frame_idx, "in_frame_set"] = True
    coord_df.to_parquet(out / "gtm_coords.parquet", index=False)

    # 6. color-coded maps -----------------------------------------------------
    t = time.time()
    figure_paths = plot_property_maps(coords, df, cfg.color_properties, out)
    timings["plot_s"] = time.time() - t

    # 7. validation -----------------------------------------------------------
    t = time.time()
    metrics: dict[str, dict] = {}
    for prop in cfg.color_properties:
        metrics[prop] = permutation_eta(
            coords,
            df[prop].to_numpy(),
            n_bins=cfg.grid_bins,
            n_perm=cfg.n_permutations,
            random_state=cfg.random_state,
        )
    metrics["crystal_system_knn_purity"] = permutation_purity(
        coords,
        coord_df["crystal_system"].to_numpy(),
        k=cfg.knn_k,
        n_perm=cfg.n_permutations,
        random_state=cfg.random_state,
    )
    timings["validate_s"] = time.time() - t

    report = {
        "dataset": cfg.dataset,
        "n_structures": int(len(df)),
        "n_frame_set": int(len(frame_idx)),
        "n_features_scm": int(X.shape[1]),
        "n_unique_spacegroups": int(df["spacegroup.number"].nunique()),
        "splits": list(cfg.splits),
        "gtm": dict(
            k=cfg.gtm_k,
            m=cfg.gtm_m,
            s=cfg.gtm_s,
            regul=cfg.gtm_regul,
            niter=cfg.gtm_niter,
        ),
        "standardize": cfg.standardize,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "metrics": metrics,
        # store figure basenames (the PNGs always sit beside report.json), so
        # the committed report stays portable instead of leaking the absolute
        # machine path the run happened to use.
        "figures": {k: Path(v).name for k, v in figure_paths.items()},
        "provenance": collect_provenance(cfg),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _format_report(report: dict) -> str:
    """Render a run report as a human-readable console summary.

    Args:
        report: The report dict returned by ``run``/``run_from_features``.

    Returns:
        A multi-line summary string.
    """
    lines = [
        f"MP-20 cartography — {report['n_structures']} structures, "
        f"{report['n_features_scm']}-dim SCM, "
        f"frame set {report['n_frame_set']}",
        "",
        "Map-quality (observed vs label-shuffled null):",
    ]
    for name, m in report["metrics"].items():
        lines.append(
            f"  {name:28s} obs={m['observed']:.3f}  "
            f"null={m['null_mean']:.3f}±{m['null_std']:.3f}  "
            f"z={m['z']:.1f}  p={m['p_value']:.4f}"
        )
    lines.append("")
    lines.append(
        "Timings (s): "
        + ", ".join(f"{k}={v}" for k, v in report["timings_s"].items())
    )
    return "\n".join(lines)
