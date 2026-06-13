"""Tests for matmercator.composition.CompositionFeaturizer."""

from __future__ import annotations

import numpy as np

from matmercator.composition import CompositionFeaturizer


def test_magpie_shape_and_finite():
    """Magpie returns a fixed 132-d finite vector per formula."""
    feat = CompositionFeaturizer()
    X = feat.transform(["NaCl", "Fe2O3", "MgO"])
    assert feat.n_features_ == 132
    assert X.shape == (3, 132)
    assert np.isfinite(X).all()


def test_deterministic():
    """Featurizing the same formulas twice gives identical vectors."""
    a = CompositionFeaturizer().transform(["LiFePO4", "SiO2"])
    b = CompositionFeaturizer().transform(["LiFePO4", "SiO2"])
    np.testing.assert_array_equal(a, b)
