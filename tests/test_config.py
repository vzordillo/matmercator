"""Tests for matmercator.config.PipelineConfig."""

from __future__ import annotations

import json
from pathlib import Path

from matmercator.config import PipelineConfig


def test_defaults_are_sane():
    """The default config carries the expected core parameters."""
    cfg = PipelineConfig()
    assert cfg.dataset == "mp_20"
    assert cfg.frame_set_size == 6000
    assert cfg.gtm_k == 16
    assert cfg.random_state == 1234
    assert tuple(cfg.splits) == ("train", "val", "test")


def test_data_root_defaults_under_data_dir():
    """The default data root points at the repo's data/ directory."""
    assert Path(PipelineConfig().data_root).name == "data"


def test_resolved_coerces_paths_and_returns_self():
    """resolved() returns the same instance with Path-typed path fields."""
    cfg = PipelineConfig(data_root="x/y", output_dir="a/b")
    out = cfg.resolved()
    assert out is cfg
    assert isinstance(cfg.data_root, Path)
    assert isinstance(cfg.output_dir, Path)


def test_to_json_roundtrips_fields(tmp_path):
    """to_json writes every field with path values stringified."""
    cfg = PipelineConfig(dataset="mp_20", gtm_k=12)
    p = tmp_path / "config.json"
    cfg.to_json(p)
    d = json.loads(p.read_text())
    assert d["dataset"] == "mp_20"
    assert d["gtm_k"] == 12
    assert isinstance(d["data_root"], str)
    assert isinstance(d["output_dir"], str)
    assert list(d["color_properties"]) == list(cfg.color_properties)
