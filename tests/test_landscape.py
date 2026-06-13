"""Tests for matmercator.landscape (node statistics, transparency, render)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from matmercator.landscape import applicability_alpha
from matmercator.landscape import coherence_penalty
from matmercator.landscape import density_alpha
from matmercator.landscape import node_statistics
from matmercator.landscape import property_landscape_panel
from matmercator.landscape import two_class_landscape
from matmercator.landscape import winning_class_landscape


def _one_hot(assignments, k):
    """Build a one-hot (n, k) responsibility matrix from node assignments."""
    R = np.zeros((len(assignments), k))
    R[np.arange(len(assignments)), assignments] = 1.0
    return R


def test_node_statistics_one_hot_exact():
    """With one-hot responsibilities, node stats reduce to per-node moments."""
    R = _one_hot([0, 0, 1, 1, 2, 2], k=3)
    values = np.array([1.0, 3.0, 10.0, 10.0, 5.0, 7.0])
    Pk, D, sigma = node_statistics(R, values)
    np.testing.assert_allclose(D, [2, 2, 2])
    np.testing.assert_allclose(Pk, [2.0, 10.0, 6.0])
    np.testing.assert_allclose(sigma, [1.0, 0.0, 1.0])


def test_node_statistics_drops_non_finite():
    """Non-finite property values (and their responsibility) are dropped."""
    R = _one_hot([0, 0, 1], k=2)
    values = np.array([2.0, np.nan, 5.0])
    Pk, D, _ = node_statistics(R, values)
    np.testing.assert_allclose(D, [1, 1])
    np.testing.assert_allclose(Pk, [2.0, 5.0])


def test_density_alpha_range_and_cutoff():
    """Opacity is in [0, 1], zero below min_count, and saturates at the top."""
    D = np.array([0.0, 1.0, 5.0, 20.0])
    a = density_alpha(D, min_count=1.0, top_pct=90.0, gamma=1.0)
    assert np.all((a >= 0) & (a <= 1))
    assert a[0] == 0.0  # below min_count
    assert np.isclose(a.max(), 1.0)


def test_coherence_penalty_thresholds():
    """Coherence penalty is 1 below sig_lo, 0 above sig_hi, linear between."""
    sigma = np.array([0.0, 0.5, 1.0, 2.0])
    pen = coherence_penalty(sigma, sig_lo=0.5, sig_hi=1.5)
    np.testing.assert_allclose(pen, [1.0, 1.0, 0.5, 0.0])


def test_applicability_is_product():
    """Applicability equals density modulation times coherence penalty."""
    D = np.array([0.0, 2.0, 8.0, 20.0])
    sigma = np.array([0.1, 0.4, 1.0, 2.0])
    expected = density_alpha(D, 1.0) * coherence_penalty(sigma, 0.5, 1.5)
    np.testing.assert_allclose(
        applicability_alpha(D, sigma, 0.5, 1.5, 1.0), expected
    )


def test_interpolated_field_is_finite_and_filled(gtm_outputs):
    """The cubic field interpolated over the node grid is finite and filled."""
    from scipy.interpolate import griddata

    from matmercator.landscape import _mesh

    R, node_coords = gtm_outputs
    values = np.random.default_rng(2).standard_normal(R.shape[0])
    Pk, _, _ = node_statistics(R, values)
    gx, gy, _ = _mesh(node_coords, res=80)
    field = griddata(node_coords, Pk, (gx, gy), method="cubic")
    # the node grid spans the map, so the cubic interpolation fills it: a
    # mostly-NaN field would mean the binning/coords path is broken.
    assert np.isfinite(field).mean() > 0.95


def test_property_landscape_panel_writes_file(gtm_outputs, tmp_path):
    """The 3-panel property landscape renders and writes a PNG."""
    R, node_coords = gtm_outputs
    values = np.random.default_rng(0).standard_normal(R.shape[0])
    out = property_landscape_panel(
        R, node_coords, values, "demo", "demo label", "viridis", tmp_path
    )
    assert Path(out).exists()
    assert out.endswith("landscape_demo.png")


def test_two_class_landscape_writes_file(gtm_outputs, tmp_path):
    """The fuzzy two-class landscape renders and writes a PNG."""
    R, node_coords = gtm_outputs
    class_values = np.random.default_rng(0).choice([1.0, 2.0], size=R.shape[0])
    out = two_class_landscape(
        R,
        node_coords,
        class_values,
        tmp_path,
        labels=("metal", "non-metal"),
        name="mc",
    )
    assert Path(out).exists()
    assert out.endswith("landscape_mc.png")


def test_winning_class_landscape_writes_file(gtm_outputs, tmp_path):
    """The discrete winning-class landscape renders and writes a PNG."""
    R, node_coords = gtm_outputs
    names = ["a", "b", "c"]
    labels = np.random.default_rng(1).choice(names, size=R.shape[0])
    out = winning_class_landscape(
        R, node_coords, labels, names, tmp_path, name="wc"
    )
    assert Path(out).exists()
    assert out.endswith("landscape_wc.png")
