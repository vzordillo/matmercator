"""Tests for matmercator.cache (feature-cache descriptor identity)."""

from __future__ import annotations

import pytest

from matmercator.cache import check_descriptor
from matmercator.cache import descriptor_spec
from matmercator.cache import write_descriptor


def test_descriptor_spec_fields():
    """The spec captures the descriptor name and both SCM settings."""
    assert descriptor_spec(diag_elems=True, sort_eigenvalues=False) == {
        "descriptor": "sine_coulomb_matrix",
        "diag_elems": True,
        "sort_eigenvalues": False,
    }


def test_write_then_check_matches(tmp_path):
    """A written sidecar validates against the same spec."""
    spec = descriptor_spec(True, True)
    write_descriptor(tmp_path, spec)
    check_descriptor(tmp_path, spec)  # must not raise


def test_check_raises_on_mismatch(tmp_path):
    """A differing descriptor raises rather than silently mixing caches."""
    write_descriptor(tmp_path, descriptor_spec(True, False))
    with pytest.raises(ValueError):
        check_descriptor(tmp_path, descriptor_spec(True, True))


def test_legacy_cache_assumed_default(tmp_path):
    """A cache without a sidecar is treated as the historical default."""
    check_descriptor(tmp_path, descriptor_spec(True, False))  # legacy default
    with pytest.raises(ValueError):
        check_descriptor(tmp_path, descriptor_spec(True, True))
