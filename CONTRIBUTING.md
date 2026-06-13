# Contributing

Early-stage research project — issues and PRs welcome.

## Setup

```bash
git clone https://github.com/vzordillo/matmercator.git
cd matmercator
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -e ".[dev]"                              # needs Python >=3.10
```

## Checks

Run before a PR — CI runs the same four:

```bash
ruff check .
ruff format --check .
mypy -p matmercator
pytest -q
```

Add or extend tests under `tests/` for any behavior change.

## Baseline

`pytest` skips `test_full_cache_regression` (the committed MP-20 baseline check)
unless the feature cache is present — it's gitignored, so a fresh clone won't
have it. To build the cache and regenerate `results/`:

```bash
matmercator features      # builds results/cache/
matmercator map           # regenerates results/mp20_scm_gtm/ (report.json + maps)
matmercator landscapes
```

Don't change the committed baseline without a measured reason; record numeric
changes in `CHANGELOG.md`.

## Notes

- The feature cache is descriptor-keyed (`src/matmercator/cache.py`) — don't mix settings.
- New descriptors and manifolds plug in alongside the SCM path
  (`src/matmercator/featurize.py`, `cartography.py`); see
  [README → Project structure](README.md#project-structure).
- Licensed MIT.
