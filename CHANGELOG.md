# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Packaging via `pyproject.toml` (`pip install -e .`) with a single
  `matmercator` console entry point and `python -m matmercator`.
- Unified, config-driven CLI (`matmercator/cli.py`) with subcommands
  `run`, `features`, `map`, `landscapes`, `hero`; JSON `--config` file plus
  flag overrides; logging to console and `run.log`.
- Package modules `featurize_cache.py`, `jobs.py`, `hero.py` (the former
  standalone scripts now delegate to the CLI as thin shims).
- Tooling: ruff (lint + format) and mypy configuration; GitHub Actions CI;
  `CONTRIBUTING.md`; this changelog.
- `sort_eigenvalues` option for the canonical (`eigh`) sorted SCM spectrum,
  with a descriptor-keyed feature cache that refuses to mix representations.
- `matmercator select` — Q²-driven (cross-validated) ranking of GTM
  hyperparameters (k, m, s, regul) by predictive power, writing
  `selection_report.json`; the keystone map-quality criterion.
- Run provenance (git SHA, dependency versions, input-CSV hashes) recorded in
  `report.json` so a map is traceable to its exact code, environment, and data.
- A composition (Magpie) descriptor (`composition.py`) and `matmercator
  experiment` — held-out Q² comparison of SCM vs composition vs union, with a
  PCA-2D baseline and a cell-size confound check (`experiment.py`).

### Changed
- Replaced the pyink/isort/pydocstyle/pylint stack with ruff for both linting
  and formatting.
- Regenerated the committed MP-20 baseline (`results/mp20_scm_gtm/report.json` +
  `RESULTS.md`) so the golden matches the current grid-η² metric, whose cell
  binning had changed since the baseline was first committed (identical
  coordinates now fall in 361 vs 372 occupied 20×20 cells). Headline metrics
  shifted slightly with unchanged conclusions — formation-energy η² 0.126→0.137,
  band gap 0.105→0.101, E-hull 0.062→0.059, crystal-system purity 0.297→0.291 —
  and `report.json` now also records run provenance.
- `report.json` stores figure file names as basenames rather than absolute
  paths, so the committed report no longer embeds the run's machine path.

## [0.1.0]

### Added
- GTM "chemical cartography" pipeline for crystal structures: CSV+CIF loader,
  Sine Coulomb Matrix featurizer, stratified frame-set fit + projection
  (`ugtm`), property point maps, and node-based property landscapes
  (density / coherence / applicability modulations, two-class and
  winning-class maps).
- Map-quality metrics (grid η² and crystal-system k-NN purity) with
  label-permutation nulls.
- MP-20 calibration baseline in `results/mp20_scm_gtm/` (45,229 structures)
  with `RESULTS.md`.
- Test suite (unit, science/methods, regression) and a committed fixture.
