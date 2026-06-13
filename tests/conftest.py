"""Shared fixtures and import-path setup for the test suite.

Inserts the ``src`` directory on ``sys.path`` so ``import matmercator`` works under
pytest regardless of the working directory, and provides a few small fixtures
reused across the heavier tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_root(repo_root: Path) -> Path:
    """Return the default dataset root (``<repo>/data``)."""
    return repo_root / "data"


@pytest.fixture(scope="session")
def fixture_root(repo_root: Path) -> Path:
    """Return the committed tiny test-fixture dataset root."""
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def tiny_structures():
    """Return a few small pymatgen Structures for featurization tests."""
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    return [
        Structure(
            Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        Structure(Lattice.cubic(3.5), ["Fe"], [[0, 0, 0]]),
        Structure(
            Lattice.cubic(5.0),
            ["Mg", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        ),
    ]


@pytest.fixture(scope="session")
def gtm_outputs():
    """Real GTM responsibilities + node grid from a small synthetic fit.

    Returns:
        A tuple ``(R, node_coords)`` with ``R`` an ``(n, K)`` responsibility
        matrix and ``node_coords`` the ``(K, 2)`` latent grid -- realistic
        inputs for the landscape render tests without needing a dataset.
    """
    from matmercator.cartography import GTMCartographer

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 8))
    carto = GTMCartographer(k=5, m=2, niter=30, random_state=0).fit(X[:60])
    return carto.responsibilities(X), carto.node_coords
