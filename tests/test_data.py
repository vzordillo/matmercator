"""Tests for matmercator.data (CIF parsing + dataset loading)."""

from __future__ import annotations

import pytest

from matmercator.data import _parse_cif
from matmercator.data import load_dataset
from matmercator.data import load_split


def test_parse_cif_roundtrip():
    """A valid CIF string parses back to a Structure with the right size."""
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    s = Structure(
        Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    parsed = _parse_cif(s.to(fmt="cif"))
    assert parsed is not None
    assert len(parsed) == 2


def test_parse_cif_returns_none_on_garbage():
    """Unparseable input yields None rather than raising."""
    assert _parse_cif("not a cif") is None


def test_load_dataset_rejects_unsupported():
    """A non-CSV dataset name raises ValueError."""
    with pytest.raises(ValueError):
        load_dataset("data", "alex_mp_20")


def test_load_split_missing_file(tmp_path):
    """A missing split CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "mp_20", "train")


def test_load_split_real_small(data_root):
    """Loading a few real MP-20 rows yields parsed structures and metadata."""
    if not (data_root / "mp_20" / "val.csv").exists():
        pytest.skip("mp_20 dataset not present")
    df = load_split(data_root, "mp_20", "val", max_structures=5)
    assert len(df) == 5
    assert "structure" in df.columns
    assert "cif" not in df.columns
    assert (df["n_sites"] > 0).all()
    assert (df["split"] == "val").all()
