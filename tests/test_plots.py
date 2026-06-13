"""Tests for matmercator.plots (crystal-system mapping + map writer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matmercator.plots import CRYSTAL_SYSTEMS
from matmercator.plots import crystal_system
from matmercator.plots import plot_property_maps


@pytest.mark.parametrize(
    "sg, system",
    [
        (1, "triclinic"),
        (2, "triclinic"),
        (3, "monoclinic"),
        (15, "monoclinic"),
        (16, "orthorhombic"),
        (74, "orthorhombic"),
        (75, "tetragonal"),
        (142, "tetragonal"),
        (143, "trigonal"),
        (167, "trigonal"),
        (168, "hexagonal"),
        (194, "hexagonal"),
        (195, "cubic"),
        (230, "cubic"),
    ],
)
def test_crystal_system_boundaries(sg, system):
    """Space-group boundary numbers map to the correct crystal system."""
    assert crystal_system(sg) == system


@pytest.mark.parametrize("sg", [0, -5, 231, 999])
def test_crystal_system_out_of_range(sg):
    """Out-of-range space-group numbers return 'unknown'."""
    assert crystal_system(sg) == "unknown"


def test_crystal_systems_seven_in_order():
    """There are exactly seven crystal systems in increasing-symmetry order."""
    assert CRYSTAL_SYSTEMS == [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic",
    ]


def test_plot_property_maps_writes_files(tmp_path):
    """plot_property_maps writes a PNG per property, plus overlay and panel."""
    rng = np.random.default_rng(0)
    n = 50
    coords = rng.standard_normal((n, 2))
    df = pd.DataFrame(
        {
            "band_gap": rng.random(n),
            "spacegroup.number": rng.integers(1, 231, n),
        }
    )
    paths = plot_property_maps(coords, df, ["band_gap"], tmp_path)
    assert set(paths) == {"band_gap", "crystal_system", "panel"}
    for p in paths.values():
        assert Path(p).exists()
