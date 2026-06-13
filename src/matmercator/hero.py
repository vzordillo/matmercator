"""Render the README hero banner from the staged feature cache.

A 1x3 panel telling the pipeline at a glance: GTM embedding (points) ->
formation-energy landscape -> crystal-system map. Uses the same config/seed as
the baseline, so the figure matches the committed run. Panel descriptions live
in the README caption, not on the figure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from matmercator import landscape as L  # noqa: E402
from matmercator.config import PipelineConfig  # noqa: E402
from matmercator.jobs import fit_cartographer_from_cache  # noqa: E402
from matmercator.plots import CRYSTAL_SYSTEMS  # noqa: E402
from matmercator.plots import crystal_system  # noqa: E402

log = logging.getLogger("matmercator")


def make_hero(cfg: PipelineConfig) -> str:
    """Render the hero banner to ``<output_dir>/hero_banner.png``.

    Args:
        cfg: Run configuration.

    Returns:
        The path to the written PNG.
    """
    cfg = cfg.resolved()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df, _, carto, R = fit_cartographer_from_cache(cfg)
    nc = carto.node_coords
    coords = R @ nc

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # 1. GTM embedding (points), colored by formation energy.
    fe = df["formation_energy_per_atom"].to_numpy()
    vlo, vhi = np.percentile(fe, [2, 98])
    axes[0].scatter(
        coords[:, 0],
        coords[:, 1],
        c=fe,
        s=3,
        alpha=0.5,
        cmap="plasma",
        vmin=vlo,
        vmax=vhi,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect("equal", "box")

    # 2. formation-energy node landscape (density-modulated), no in-figure title.
    Pk, D, _ = L.node_statistics(R, fe)
    vmn, vmx = np.percentile(Pk[D >= 1.0], [3, 97])
    L._render_continuous(
        axes[1], nc, Pk, L.density_alpha(D), "plasma", vmn, vmx, ""
    )

    # 3. crystal-system winning-class map, legend moved INSIDE the panel.
    csys = df["spacegroup.number"].map(crystal_system).to_numpy()
    score = np.zeros((R.shape[1], len(CRYSTAL_SYSTEMS)))
    for j, name in enumerate(CRYSTAL_SYSTEMS):
        msk = csys == name
        if msk.any():
            score[:, j] = R[msk].sum(axis=0)
    L._render_discrete(
        axes[2],
        nc,
        score.argmax(axis=1),
        L.density_alpha(R.sum(axis=0)),
        CRYSTAL_SYSTEMS,
    )
    leg = axes[2].get_legend()
    if leg is not None:
        leg.remove()
    handles = [
        Patch(color=plt.get_cmap("tab10")(i), label=name)
        for i, name in enumerate(CRYSTAL_SYSTEMS)
    ]
    axes[2].legend(
        handles=handles,
        fontsize=6,
        loc="lower left",
        ncol=2,
        framealpha=0.85,
        handlelength=1.0,
        borderpad=0.4,
        columnspacing=1.0,
    )

    fig.tight_layout()
    path = out / "hero_banner.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved hero banner -> %s", path)
    return str(path)
