"""Feature-cache descriptor identity: write and validate.

The staged path caches descriptor vectors (``X_{split}.npz``) whose values
depend on the descriptor settings (``diag_elems``, ``sort_eigenvalues``). The
filenames do not encode those settings, so building the cache one way and then
building the map with a different config would silently mix representations.

To prevent that, ``build_features.py`` writes a small ``descriptor.json`` sidecar
recording the descriptor identity, and the map/landscape builders validate it
against their config before loading -- raising on a mismatch rather than
proceeding. A legacy cache without the sidecar is treated as the historical
default (``diag_elems=True``, ``sort_eigenvalues=False``).
"""

from __future__ import annotations

import json
from pathlib import Path

DESCRIPTOR_FILE = "descriptor.json"


def descriptor_spec(diag_elems: bool, sort_eigenvalues: bool) -> dict:
    """Return the descriptor-identity dict stored alongside the features.

    Args:
        diag_elems: Whether the SCM diagonal self-terms are included.
        sort_eigenvalues: Whether the canonical (eigh) sorted spectrum is used.

    Returns:
        A JSON-serializable dict identifying the descriptor.
    """
    return {
        "descriptor": "sine_coulomb_matrix",
        "diag_elems": bool(diag_elems),
        "sort_eigenvalues": bool(sort_eigenvalues),
    }


def write_descriptor(cache_dir: str | Path, spec: dict) -> None:
    """Write the descriptor sidecar into ``cache_dir``.

    Args:
        cache_dir: Directory holding the feature cache.
        spec: Descriptor identity from :func:`descriptor_spec`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / DESCRIPTOR_FILE).write_text(json.dumps(spec, indent=2))


def check_descriptor(cache_dir: str | Path, expected: dict) -> None:
    """Validate the cache's descriptor matches ``expected``.

    Args:
        cache_dir: Directory holding the feature cache.
        expected: Descriptor identity the current run requires.

    Raises:
        ValueError: If the cache's descriptor does not match ``expected``.
    """
    path = Path(cache_dir) / DESCRIPTOR_FILE
    if path.exists():
        found = json.loads(path.read_text())
    else:
        # legacy cache predating the sidecar -> historical default
        found = descriptor_spec(diag_elems=True, sort_eigenvalues=False)
    if found != expected:
        raise ValueError(
            f"feature cache descriptor {found} does not match the run config "
            f"{expected}; rebuild the cache with scripts/build_features.py "
            "using matching settings."
        )
