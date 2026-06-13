"""Tests for matmercator.featurize.SCMFeaturizer."""

from __future__ import annotations

import numpy as np
import pytest

from matmercator.featurize import SCMFeaturizer


def test_transform_before_fit_raises(tiny_structures):
    """Calling transform before fit raises RuntimeError."""
    feat = SCMFeaturizer()
    with pytest.raises(RuntimeError):
        feat.transform(tiny_structures)


def test_fit_transform_shape_and_padding(tiny_structures):
    """Features form an (n, max_atoms) array and set n_features_."""
    feat = SCMFeaturizer()
    X = feat.fit_transform(tiny_structures)
    max_atoms = max(len(s) for s in tiny_structures)
    assert X.shape == (len(tiny_structures), max_atoms)
    assert feat.n_features_ == max_atoms
    assert np.isfinite(X).all()


def test_featurization_is_deterministic(tiny_structures):
    """Featurizing the same structures twice gives identical vectors."""
    a = SCMFeaturizer().fit_transform(tiny_structures)
    b = SCMFeaturizer().fit_transform(tiny_structures)
    np.testing.assert_array_equal(a, b)


def test_permutation_invariance(tiny_structures):
    """The SCM eigenvalue spectrum is invariant to atom (site) ordering.

    matminer returns the eigenvalues unsorted, so the invariant object is the
    spectrum (the multiset of eigenvalues), compared here after sorting.
    """
    from pymatgen.core import Structure

    feat = SCMFeaturizer().fit(tiny_structures)
    s = tiny_structures[2]  # 3-site cell -> a non-trivial reordering
    order = [2, 0, 1]
    reordered = Structure(
        s.lattice,
        [s.species[i] for i in order],
        [s.frac_coords[i] for i in order],
    )
    np.testing.assert_allclose(
        np.sort(feat.transform([s]).ravel()),
        np.sort(feat.transform([reordered]).ravel()),
        atol=1e-6,
    )


def test_translation_invariance(tiny_structures):
    """The SCM eigenvalue spectrum is invariant to a global translation."""
    from pymatgen.core import Structure

    feat = SCMFeaturizer().fit(tiny_structures)
    s = tiny_structures[2]
    shifted = Structure(s.lattice, s.species, s.frac_coords + 0.137)
    np.testing.assert_allclose(
        np.sort(feat.transform([s]).ravel()),
        np.sort(feat.transform([shifted]).ravel()),
        atol=1e-6,
    )


def test_sort_eigenvalues_canonical(tiny_structures):
    """sort_eigenvalues=True gives a descending, canonical, invariant vector.

    The symmetric-solver spectrum equals the unsorted one as a set, comes out
    sorted descending, and is element-wise permutation-invariant (stronger than
    the default's spectrum-only invariance).
    """
    from pymatgen.core import Structure

    raw = SCMFeaturizer(sort_eigenvalues=False).fit(tiny_structures)
    srt = SCMFeaturizer(sort_eigenvalues=True).fit(tiny_structures)
    s = tiny_structures[2]  # 3-site cell
    k = len(s)

    vec = srt.transform([s])[0]
    assert np.all(np.diff(vec[:k]) <= 1e-9)  # descending over real eigenvalues
    np.testing.assert_allclose(
        np.sort(vec), np.sort(raw.transform([s])[0]), atol=1e-6
    )

    order = [2, 0, 1]
    reordered = Structure(
        s.lattice,
        [s.species[i] for i in order],
        [s.frac_coords[i] for i in order],
    )
    np.testing.assert_allclose(
        srt.transform([s]), srt.transform([reordered]), atol=1e-6
    )
