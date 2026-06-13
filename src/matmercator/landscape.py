"""Node-based property landscapes (a standard GTM technique).

The point scatter in ``plots.py`` is the responsibility *barycenter* of each
structure -- the least informative view. The richer object is the landscape
defined on the K latent nodes from the full responsibility matrix R (n x K):

    density       D_k     = sum_n R_kn                    (fuzzy resident count)
    mean property P_k     = sum_n R_kn w_n P_n / sum_n R_kn w_n
    coherence     sigma_k = sqrt( <P^2>_k - P_k^2 )       (resp.-weighted spread)

P_k is interpolated into a smooth field over the latent square; transparency
encodes trustworthiness, giving three complementary views:

    a) density-modulated       alpha ~ D_k          (empty zones fade out)
    b) coherence-modulated     alpha ~ low sigma_k  (incoherent zones fade out)
    c) applicability-modulated alpha ~ density x coherence-penalty

Materials properties span very different scales, so the sigma thresholds default
to multiples of the global property spread (sigma_hi = std(P), sigma_lo = 1/3
std(P)); override per property if you have a physical tolerance in mind.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / file output; set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.interpolate import griddata

__all__ = [
    "node_statistics",
    "property_landscape_panel",
    "two_class_landscape",
    "winning_class_landscape",
]


# --------------------------------------------------------------------------- #
# node-level statistics
# --------------------------------------------------------------------------- #
def node_statistics(
    R: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-node density, mean and coherence from responsibilities.

    Uses matmuls against R (no n x K temporaries beyond R itself), so it scales
    to the full corpus. Non-finite property values are dropped.

    Args:
        R: ``(n, K)`` responsibility matrix.
        values: ``(n,)`` property values aligned to ``R``'s rows.
        weights: Optional ``(n,)`` per-structure weights (default: all ones).

    Returns:
        A tuple ``(P_k, D_k, sigma_k)`` of per-node mean property, density
        (fuzzy resident count) and responsibility-weighted coherence, each of
        shape ``(K,)``.
    """
    R = np.asarray(R, dtype=float)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    R, v = R[finite], v[finite]
    w = (
        np.ones(len(v))
        if weights is None
        else np.asarray(weights, float)[finite]
    )

    D = R.sum(axis=0)  # (K,) density, unweighted
    denom = np.maximum(R.T @ w, 1e-12)  # (K,) floor avoids /0 on empty nodes
    Pk = (R.T @ (w * v)) / denom
    ex2 = (R.T @ (w * v * v)) / denom
    # clip to 0 guards a tiny negative variance from floating-point rounding
    sigma = np.sqrt(np.clip(ex2 - Pk**2, 0.0, None))
    return Pk, D, sigma


# --------------------------------------------------------------------------- #
# transparency modulations
# --------------------------------------------------------------------------- #
def density_alpha(
    D: np.ndarray,
    min_count: float = 1.0,
    top_pct: float = 90.0,
    gamma: float = 0.6,
) -> np.ndarray:
    """Opacity from cumulated responsibility (node density).

    Nodes at/above the ``top_pct`` density reach full color and fainter ones
    fade below; ``gamma`` < 1 lifts mid densities so populated regions stay
    vivid. Nodes below ``min_count`` are forced fully transparent.

    Args:
        D: ``(K,)`` per-node density.
        min_count: Hard transparency cutoff, in cumulated-responsibility units
            (fuzzy resident count) -- "practically empty" nodes.
        top_pct: Density percentile mapped to full opacity.
        gamma: Gamma correction (< 1 lifts mid densities).

    Returns:
        A ``(K,)`` array of opacities in ``[0, 1]``.
    """
    D = np.asarray(D, dtype=float)
    ref = max(np.percentile(D, top_pct), 1e-12)  # floor avoids /0 if all empty
    a = np.clip(D / ref, 0.0, 1.0) ** gamma
    a[min_count > D] = 0.0
    return a


def coherence_penalty(
    sigma: np.ndarray, sig_lo: float, sig_hi: float
) -> np.ndarray:
    """Transparency from coherence: 1 where coherent, 0 where incoherent.

    Args:
        sigma: ``(K,)`` per-node coherence (spread).
        sig_lo: Spread at/below which a node is fully opaque.
        sig_hi: Spread at/above which a node is fully transparent.

    Returns:
        A ``(K,)`` array in ``[0, 1]`` (1 where ``sigma <= sig_lo``, 0 where
        ``sigma >= sig_hi``, linear in between).
    """
    return np.clip((sig_hi - sigma) / max(sig_hi - sig_lo, 1e-12), 0.0, 1.0)


def applicability_alpha(
    D: np.ndarray,
    sigma: np.ndarray,
    sig_lo: float,
    sig_hi: float,
    min_count: float = 1.0,
) -> np.ndarray:
    """Combined opacity = density modulation x coherence penalty.

    Args:
        D: ``(K,)`` per-node density.
        sigma: ``(K,)`` per-node coherence (spread).
        sig_lo: Spread at/below which a node is fully coherent.
        sig_hi: Spread at/above which a node is fully incoherent.
        min_count: Density cutoff forwarded to ``density_alpha``.

    Returns:
        A ``(K,)`` array of opacities in ``[0, 1]``.
    """
    return density_alpha(D, min_count) * coherence_penalty(
        sigma, sig_lo, sig_hi
    )


# --------------------------------------------------------------------------- #
# rendering primitives
# --------------------------------------------------------------------------- #
def _mesh(node_coords: np.ndarray, res: int = 220):
    """Build a ``res x res`` interpolation grid spanning the node coordinates.

    Args:
        node_coords: ``(K, 2)`` latent node coordinates.
        res: Grid resolution per axis.

    Returns:
        A tuple ``(gx, gy, extent)`` where ``gx``/``gy`` are meshgrids and
        ``extent`` is ``(xmin, xmax, ymin, ymax)`` for ``imshow``.
    """
    xmin, xmax = node_coords[:, 0].min(), node_coords[:, 0].max()
    ymin, ymax = node_coords[:, 1].min(), node_coords[:, 1].max()
    gx, gy = np.meshgrid(
        np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res)
    )
    return gx, gy, (xmin, xmax, ymin, ymax)


def _render_continuous(
    ax,
    node_coords,
    Pk,
    alpha,
    cmap,
    vmin,
    vmax,
    title,
    res=220,
    show_nodes=True,
):
    """Interpolate scattered node values onto a fine mesh and render RGBA.

    Uses ``griddata`` (cubic for the property field so it reads smoothly, linear
    for the alpha field to avoid overshoot) over the actual node coordinates --
    robust to the node layout, with no lattice assumption. The latent nodes sit
    at the corners of the square, so the interpolation covers the whole map.

    Args:
        ax: Matplotlib axis to draw on.
        node_coords: ``(K, 2)`` latent node coordinates.
        Pk: ``(K,)`` per-node property values.
        alpha: ``(K,)`` per-node opacities.
        cmap: Matplotlib colormap name.
        vmin: Lower color limit.
        vmax: Upper color limit.
        title: Axis title.
        res: Interpolation grid resolution per axis.
        show_nodes: Overlay the node positions as small circles.

    Returns:
        A ``ScalarMappable`` carrying the norm + colormap, for a shared
        colorbar.
    """
    cmap = plt.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    gx, gy, extent = _mesh(node_coords, res)

    Pf = griddata(node_coords, Pk, (gx, gy), method="cubic")
    Af = griddata(node_coords, np.clip(alpha, 0, 1), (gx, gy), method="linear")
    Pf = np.where(np.isfinite(Pf), Pf, vmin)
    Af = np.clip(np.nan_to_num(Af, nan=0.0), 0, 1)

    rgba = cmap(norm(Pf))
    rgba[..., 3] = Af
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )

    if show_nodes:  # the 'circles cut out of the landscape'
        ncol = cmap(norm(Pk))
        ncol[:, 3] = np.clip(alpha, 0, 1)
        ax.scatter(
            node_coords[:, 0], node_coords[:, 1], c=ncol, s=4, linewidths=0
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return ScalarMappable(norm=norm, cmap=cmap)


def _render_discrete(ax, node_coords, node_class, alpha, class_names, res=220):
    """Render a discrete per-node class field with nearest-neighbour fill.

    Args:
        ax: Matplotlib axis to draw on.
        node_coords: ``(K, 2)`` latent node coordinates.
        node_class: ``(K,)`` integer class index per node.
        alpha: ``(K,)`` per-node opacities.
        class_names: Class names for the legend (index-aligned).
        res: Interpolation grid resolution per axis.
    """
    cmap = plt.get_cmap("tab10")
    gx, gy, extent = _mesh(node_coords, res)
    Cf = griddata(
        node_coords, node_class.astype(float), (gx, gy), method="nearest"
    )
    Af = griddata(node_coords, np.clip(alpha, 0, 1), (gx, gy), method="linear")
    Af = np.clip(np.nan_to_num(Af, nan=0.0), 0, 1)
    rgba = cmap(np.clip(np.rint(Cf).astype(int), 0, 9))
    rgba[..., 3] = Af
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        Patch(color=cmap(i), label=name) for i, name in enumerate(class_names)
    ]
    ax.legend(
        handles=handles,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        framealpha=0.9,
    )


# --------------------------------------------------------------------------- #
# public landscape builders
# --------------------------------------------------------------------------- #
def property_landscape_panel(
    R: np.ndarray,
    node_coords: np.ndarray,
    values: np.ndarray,
    name: str,
    label: str,
    cmap: str,
    output_dir: str | Path,
    *,
    clip=(2, 98),
    min_count: float = 1.0,
    sig_lo: float | None = None,
    sig_hi: float | None = None,
) -> str:
    """Write a 3-panel density / coherence / applicability landscape.

    Args:
        R: ``(n, K)`` responsibility matrix.
        node_coords: ``(K, 2)`` latent node coordinates.
        values: ``(n,)`` property values.
        name: File stem (writes ``landscape_{name}.png``).
        label: Human-readable property label for the colorbar/title.
        cmap: Matplotlib colormap name.
        output_dir: Directory to write the PNG to (created if needed).
        clip: Percentile range for the color limits.
        min_count: Density cutoff for transparency.
        sig_lo: Lower coherence threshold (default: 1/3 of the property std).
        sig_hi: Upper coherence threshold (default: the property std).

    Returns:
        The path of the written PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Pk, D, sigma = node_statistics(R, values)

    # Color scale from the NODE-MEAN distribution actually shown (density-gated),
    # not the raw per-structure range: averaging shrinks node means inward, so a
    # raw-range scale would leave the map stuck in the colormap's middle.
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    valid = min_count <= D
    ref_pk = Pk[valid] if valid.any() else Pk
    vmin, vmax = np.percentile(ref_pk, clip)
    gstd = float(np.std(v[finite]))  # property scale for sigma thresholds
    sig_lo = (gstd / 3.0) if sig_lo is None else sig_lo
    sig_hi = gstd if sig_hi is None else sig_hi

    variants = {
        "density": density_alpha(D, min_count),
        "coherence": coherence_penalty(sigma, sig_lo, sig_hi),
        "applicability": applicability_alpha(
            D, sigma, sig_lo, sig_hi, min_count
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    sm = None
    for ax, (mod, a) in zip(axes, variants.items(), strict=False):
        sm = _render_continuous(
            ax, node_coords, Pk, a, cmap, vmin, vmax, f"{mod}-modulated"
        )
    fig.suptitle(label, y=1.0, fontsize=12)
    assert sm is not None
    cb = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label(label)
    p = output_dir / f"landscape_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def two_class_landscape(
    R: np.ndarray,
    node_coords: np.ndarray,
    class_values: np.ndarray,
    output_dir: str | Path,
    *,
    labels: Sequence[str] = ("class 1", "class 2"),
    min_count: float = 1.0,
    name: str = "two_class",
) -> str:
    """Write a fuzzy two-class landscape (P_k in [1, 2], red<->blue).

    Args:
        R: ``(n, K)`` responsibility matrix.
        node_coords: ``(K, 2)`` latent node coordinates.
        class_values: ``(n,)`` per-structure class code (1.0 or 2.0).
        output_dir: Directory to write the PNG to (created if needed).
        labels: ``(class 1, class 2)`` display names for the colorbar ends.
        min_count: Density cutoff for transparency.
        name: File stem (writes ``landscape_{name}.png``).

    Returns:
        The path of the written PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Pk, D, _ = node_statistics(R, class_values)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sm = _render_continuous(
        ax,
        node_coords,
        Pk,
        density_alpha(D, min_count),
        "RdYlBu",
        1.0,
        2.0,
        f"{labels[0]} (red) ↔ {labels[1]} (blue)",
    )
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=[1, 1.5, 2])
    cb.ax.set_yticklabels([labels[0], "mixed", labels[1]])
    p = output_dir / f"landscape_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def winning_class_landscape(
    R: np.ndarray,
    node_coords: np.ndarray,
    labels_array: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    *,
    min_count: float = 1.0,
    name: str = "winning_class",
) -> str:
    """Write a discrete landscape coloring each node by its winning class.

    Each node is assigned the class with the highest cumulated responsibility.

    Args:
        R: ``(n, K)`` responsibility matrix.
        node_coords: ``(K, 2)`` latent node coordinates.
        labels_array: ``(n,)`` per-structure class labels.
        class_names: The full set of class names (index order is the legend).
        output_dir: Directory to write the PNG to (created if needed).
        min_count: Density cutoff for transparency.
        name: File stem (writes ``landscape_{name}.png``).

    Returns:
        The path of the written PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    L = np.asarray(labels_array)
    D = R.sum(axis=0)
    score = np.zeros((R.shape[1], len(class_names)))
    for j, c in enumerate(class_names):
        m = np.asarray(c == L)
        if m.any():
            score[:, j] = R[m].sum(axis=0)
    node_class = score.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    _render_discrete(
        ax, node_coords, node_class, density_alpha(D, min_count), class_names
    )
    ax.set_title("winning class (density-modulated)", fontsize=10)
    p = output_dir / f"landscape_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(p)
