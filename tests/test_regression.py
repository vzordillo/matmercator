"""Regression guards that pin the pipeline's scientific outputs.

Two layers (per the test plan):
  * a small, hermetic run on the committed fixture that ALWAYS runs and checks
    determinism + null calibration + the output contract;
  * a full-scale run on the committed feature cache that reproduces the headline
    metrics in ``results/mp20_scm_gtm/report.json``, skipped if the cache is
    absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from matmercator.config import PipelineConfig
from matmercator.metrics import _bin_index
from matmercator.pipeline import run
from matmercator.pipeline import run_from_features


def _fixture_cfg(out, fixture_root):
    """Small, deterministic config over the committed fixture dataset."""
    return PipelineConfig(
        data_root=fixture_root,
        dataset="mp_20",
        splits=("train", "val"),
        frame_set_size=25,
        gtm_k=4,
        gtm_niter=40,
        n_permutations=100,
        output_dir=out,
    )


def test_small_run_is_reproducible_and_calibrated(fixture_root, tmp_path):
    """The fixture run is deterministic, calibrated, and writes its artifacts."""
    rep = run(_fixture_cfg(tmp_path / "a", fixture_root))
    run(_fixture_cfg(tmp_path / "b", fixture_root))

    cols = ["gtm_x", "gtm_y"]
    da = pd.read_parquet(tmp_path / "a" / "gtm_coords.parquet")
    db = pd.read_parquet(tmp_path / "b" / "gtm_coords.parquet")

    # determinism: identical coordinates across independent runs
    np.testing.assert_allclose(da[cols].to_numpy(), db[cols].to_numpy())
    coords = da[cols].to_numpy()
    assert np.isfinite(coords).all()
    assert int(da["in_frame_set"].sum()) == rep["n_frame_set"]

    for m in rep["metrics"].values():
        assert np.isfinite(m["observed"])
        assert 0 < m["p_value"] <= 1

    # null calibration holds at any n: null_mean ~ (n_occupied - 1)/(n - 1)
    n = len(coords)
    g = len(np.unique(_bin_index(coords, 20)))
    floor = (g - 1) / (n - 1)
    assert np.isclose(rep["metrics"]["band_gap"]["null_mean"], floor, rtol=0.2)

    # artifacts written
    cfg_json = json.loads((tmp_path / "a" / "config.json").read_text())
    assert cfg_json["dataset"] == "mp_20"
    assert (tmp_path / "a" / "report.json").exists()
    assert (tmp_path / "a" / "map_panel.png").exists()


def test_full_cache_regression(repo_root, tmp_path):
    """Re-running the map on the committed cache reproduces report.json."""
    cache = repo_root / "results" / "cache"
    report_path = repo_root / "results" / "mp20_scm_gtm" / "report.json"
    if not (cache / "X_train.npz").exists() or not report_path.exists():
        pytest.skip("feature cache or golden report not present")
    expected = json.loads(report_path.read_text())

    cfg = PipelineConfig(output_dir=tmp_path).resolved()
    metas, xs = [], []
    for split in cfg.splits:
        metas.append(pd.read_parquet(cache / f"meta_{split}.parquet"))
        xs.append(np.load(cache / f"X_{split}.npz")["X"])
    df = pd.concat(metas, ignore_index=True)
    X = np.vstack(xs).astype(float)
    X = X[:, np.abs(X).sum(axis=0) > 0]  # drop zero-pad columns

    report = run_from_features(cfg, df, X)

    assert report["n_structures"] == expected["n_structures"] == 45229
    assert report["n_features_scm"] == expected["n_features_scm"] == 20

    # headline magnitudes reproduce the golden within a loose scientific band
    for name, golden in expected["metrics"].items():
        got = report["metrics"][name]
        np.testing.assert_allclose(
            got["observed"], golden["observed"], rtol=0.05
        )
        np.testing.assert_allclose(
            got["null_mean"], golden["null_mean"], rtol=0.1
        )
        assert got["z"] > 50  # committed z are 84-228

    # the continuous nulls equal the analytic ANOVA floor (n_occupied-1)/(n-1)
    coords = pd.read_parquet(tmp_path / "gtm_coords.parquet")
    n = report["n_structures"]
    g = len(np.unique(_bin_index(coords[["gtm_x", "gtm_y"]].to_numpy(), 20)))
    floor = (g - 1) / (n - 1)
    for name in ("band_gap", "formation_energy_per_atom", "e_above_hull"):
        assert np.isclose(report["metrics"][name]["null_mean"], floor, rtol=0.1)
