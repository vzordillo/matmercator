"""Color-code the GTM landscape by physical properties.

Each plot is a scatter of the 2-D GTM coordinates, one point per structure,
colored by a property. Continuous properties use percentile clipping on the
color scale (default 2nd-98th) so a few outliers don't wash out the gradient;
the clip range is annotated. The space-group label is collapsed to the seven
crystal systems for an interpretable categorical overlay.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output; set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np

# International Tables space-group-number ranges -> crystal system.
_CRYSTAL_SYSTEM_BOUNDS = [
    (1, 2, "triclinic"),
    (3, 15, "monoclinic"),
    (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"),
    (143, 167, "trigonal"),
    (168, 194, "hexagonal"),
    (195, 230, "cubic"),
]
CRYSTAL_SYSTEMS = [name for *_, name in _CRYSTAL_SYSTEM_BOUNDS]

# Human-readable axis labels + colormap per property.
_PROP_META = {
    "band_gap": ("band gap (eV)", "viridis"),
    "formation_energy_per_atom": ("formation energy (eV/atom)", "coolwarm"),
    "e_above_hull": ("E above hull (eV/atom)", "magma"),
}


def crystal_system(sg_number: int) -> str:
    """Map a space-group number to its crystal system.

    Args:
        sg_number: International Tables space-group number (1-230).

    Returns:
        The crystal-system name, or ``"unknown"`` if out of range.
    """
    for lo, hi, name in _CRYSTAL_SYSTEM_BOUNDS:
        if lo <= sg_number <= hi:
            return name
    return "unknown"


def _scatter(ax, coords, values, title, *, clip=(2, 98), cmap="viridis"):
    """Scatter one property over the map onto ``ax``.

    Args:
        ax: Matplotlib axis to draw on.
        coords: ``(n, 2)`` map coordinates.
        values: ``(n,)`` property values colored on the points.
        title: Axis title and colorbar label.
        clip: Percentile range for the color limits (robust to outliers).
        cmap: Matplotlib colormap name.

    Returns:
        The scatter ``PathCollection``.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    vmin, vmax = (
        np.percentile(v[finite], clip) if finite.any() else (None, None)
    )
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=v,
        s=4,
        alpha=0.6,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
    )
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(title)
    ax.set_title(title)
    ax.set_xlabel("GTM latent x")
    ax.set_ylabel("GTM latent y")
    ax.set_aspect("equal", "box")
    return sc


def plot_property_maps(
    coords, df, properties: Sequence[str], output_dir
) -> dict[str, str]:
    """Write one PNG per property plus a combined panel.

    Args:
        coords: ``(n, 2)`` map coordinates.
        df: DataFrame row-aligned to ``coords``; must hold each property column
            and ``spacegroup.number``.
        properties: Continuous property column names to plot.
        output_dir: Directory the PNGs are written to (created if needed).

    Returns:
        A mapping ``{name: path}`` for each property, ``"crystal_system"`` and
        the combined ``"panel"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for prop in properties:
        label, cmap = _PROP_META.get(prop, (prop, "viridis"))
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        _scatter(ax, coords, df[prop].to_numpy(), label, cmap=cmap)
        fig.tight_layout()
        p = output_dir / f"map_{prop}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths[prop] = str(p)

    # crystal-system categorical overlay
    systems = df["spacegroup.number"].map(crystal_system).to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    cmap_obj = plt.get_cmap("tab10")
    for i, name in enumerate(CRYSTAL_SYSTEMS):
        m = systems == name
        if m.any():
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=4,
                alpha=0.6,
                color=cmap_obj(i),
                label=name,
                linewidths=0,
                rasterized=True,
            )
    ax.set_title("crystal system")
    ax.set_xlabel("GTM latent x")
    ax.set_ylabel("GTM latent y")
    ax.set_aspect("equal", "box")
    ax.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    p = output_dir / "map_crystal_system.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["crystal_system"] = str(p)

    # combined panel
    n = len(properties) + 1
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, prop in zip(axes, properties, strict=False):
        label, cmap = _PROP_META.get(prop, (prop, "viridis"))
        _scatter(ax, coords, df[prop].to_numpy(), label, cmap=cmap)
    ax = axes[len(properties)]
    cmap_obj = plt.get_cmap("tab10")
    for i, name in enumerate(CRYSTAL_SYSTEMS):
        m = systems == name
        if m.any():
            ax.scatter(
                coords[m, 0],
                coords[m, 1],
                s=4,
                alpha=0.6,
                color=cmap_obj(i),
                label=name,
                linewidths=0,
                rasterized=True,
            )
    ax.set_title("crystal system")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("GTM latent x")
    ax.set_ylabel("GTM latent y")
    ax.legend(markerscale=3, fontsize=7, loc="best")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    p = output_dir / "map_panel.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["panel"] = str(p)
    return paths
