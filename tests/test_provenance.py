"""Tests for matmercator.provenance.collect_provenance."""

from __future__ import annotations

from matmercator.config import PipelineConfig
from matmercator.provenance import collect_provenance


def test_provenance_has_expected_shape(fixture_root):
    """Provenance records versions and hashes the present input CSVs."""
    cfg = PipelineConfig(
        data_root=fixture_root, dataset="mp_20", splits=("train", "val")
    )
    prov = collect_provenance(cfg)

    assert set(prov) == {"git_sha", "versions", "inputs"}
    # numpy is always installed in the test environment
    assert "numpy" in prov["versions"]
    # the fixture ships train/val CSVs, which get hashed
    assert "train" in prov["inputs"]
    entry = prov["inputs"]["train"]
    assert entry["file"] == "mp_20/train.csv"
    assert len(entry["sha256_16"]) == 16
    assert entry["bytes"] > 0
