"""Run provenance: git SHA, dependency versions, input-file hashes.

Recorded in ``report.json`` so a given map is traceable to the exact code,
environment, and input data that produced it. Pairs with the pinned
``requirements.txt`` / ``pyproject.toml`` to make a run reproducible.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import subprocess
from importlib import metadata
from pathlib import Path

from matmercator.config import PipelineConfig

log = logging.getLogger("matmercator")

_PKGS = [
    "matmercator",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "pymatgen",
    "matminer",
    "ugtm",
    "spglib",
]


def _git_sha() -> str | None:
    """Return the short git commit hash, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _versions() -> dict[str, str]:
    """Resolved versions of the key runtime packages that are installed."""
    versions: dict[str, str] = {}
    for pkg in _PKGS:
        with contextlib.suppress(Exception):
            versions[pkg] = metadata.version(pkg)
    return versions


def _sha256_16(path: Path) -> str:
    """First 16 hex chars of a file's SHA-256 (enough to detect changes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def collect_provenance(cfg: PipelineConfig) -> dict:
    """Gather git SHA, package versions, and input-CSV hashes for a run.

    Args:
        cfg: The run configuration.

    Returns:
        A JSON-serializable dict with keys ``git_sha``, ``versions`` and
        ``inputs`` (per-split dataset-CSV hash/size, when the CSVs are present).
    """
    cfg = cfg.resolved()
    inputs: dict[str, dict] = {}
    for split in cfg.splits:
        csv = Path(cfg.data_root) / cfg.dataset / f"{split}.csv"
        if csv.exists():
            inputs[split] = {
                "file": f"{cfg.dataset}/{split}.csv",
                "sha256_16": _sha256_16(csv),
                "bytes": csv.stat().st_size,
            }
    return {"git_sha": _git_sha(), "versions": _versions(), "inputs": inputs}
