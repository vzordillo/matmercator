"""Parallel SCM feature-cache builder.

Parsing tens of thousands of CIFs is single-thread-bound, so this module
featurizes each split across multiple processes and writes a reusable cache
(``X_{split}.npz`` + ``meta_{split}.parquet`` + ``descriptor.json``) under the
cache directory. The staged CLI subcommands (``map``, ``landscapes``, ``hero``)
then read that cache instead of re-parsing the CIFs.

Each CIF becomes a Sine Coulomb Matrix eigenvalue vector zero-padded to
``MAX_ATOMS`` columns (a fixed width so every worker/split agrees; unused
trailing columns are dropped when the cache is assembled). The descriptor
identity (``diag_elems``, ``sort_eigenvalues``) is written beside the cache so
the map builders refuse to mix representations.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matmercator.cache import descriptor_spec
from matmercator.cache import write_descriptor
from matmercator.config import PipelineConfig

log = logging.getLogger("matmercator")

MAX_ATOMS = 24  # safe upper bound for MP-20 (cells are <= 20 atoms)
_PROPS = ["formation_energy_per_atom", "band_gap", "e_above_hull"]

_SCM: Any = None  # per-worker featurizer, set in the initializer
_SORT = False  # per-worker copy of the sort_eigenvalues flag


def cache_dir_for(cfg: PipelineConfig) -> Path:
    """Return the cache directory for a config (beside ``output_dir``)."""
    return Path(cfg.output_dir).resolve().parent / "cache"


def _init_worker(diag_elems: bool, sort_eigenvalues: bool) -> None:
    """Build the per-worker featurizer once, fixing the pad length.

    Fits on a dummy ``MAX_ATOMS``-site cell so the eigenvalue vector is padded
    to a fixed length in every worker; the dummy only sets the pad length, real
    eigenvalues are computed per structure and independent of it.

    Args:
        diag_elems: Include the diagonal SCM self-terms.
        sort_eigenvalues: Use the canonical symmetric-solver sorted spectrum.
    """
    global _SCM, _SORT
    _SORT = sort_eigenvalues
    warnings.filterwarnings("ignore")
    from matminer.featurizers.structure import SineCoulombMatrix
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    scm = SineCoulombMatrix(diag_elems=diag_elems, flatten=not sort_eigenvalues)
    scm.set_n_jobs(1)
    dummy = Structure(
        Lattice.cubic(10.0),
        ["H"] * MAX_ATOMS,
        np.random.default_rng(0).random((MAX_ATOMS, 3)),
    )
    scm.fit([dummy])
    _SCM = scm


def _feat_one(cif: str):
    """Featurize one CIF string in a worker.

    Returns:
        ``(vec, n_sites)`` with ``vec`` a ``float32`` length-``MAX_ATOMS``
        eigenvalue vector, ``("TOOBIG", n)`` if the cell exceeds ``MAX_ATOMS``,
        or ``None`` if parsing fails.
    """
    from pymatgen.core import Structure

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Structure.from_str(cif, fmt="cif")
    except Exception:
        return None
    n = len(s)
    if n > MAX_ATOMS:
        return ("TOOBIG", n)
    if _SORT:
        mat = np.asarray(_SCM.featurize(s)[0], dtype=np.float64)
        eigs = np.linalg.eigvalsh(mat)[::-1]  # real, descending
        vec = np.zeros(MAX_ATOMS, dtype=np.float32)
        vec[: len(eigs)] = eigs
        return vec, n
    return np.asarray(_SCM.featurize(s), dtype=np.float32), n


def build_cache_split(
    cfg: PipelineConfig,
    split: str,
    *,
    procs: int = 4,
    out_dir: Path | None = None,
    max_structures: int | None = None,
) -> Path:
    """Featurize one split in parallel and write its cache shards.

    Args:
        cfg: Run configuration (supplies ``data_root``, ``dataset`` and the
            descriptor settings).
        split: Split name (``train`` / ``val`` / ``test``).
        procs: Number of worker processes.
        out_dir: Cache directory (defaults to :func:`cache_dir_for`).
        max_structures: Optional cap on rows (for quick tests).

    Returns:
        The cache directory the shards were written to.

    Raises:
        RuntimeError: If any structure exceeds ``MAX_ATOMS``.
    """
    cfg = cfg.resolved()
    out = Path(out_dir) if out_dir is not None else cache_dir_for(cfg)
    out.mkdir(parents=True, exist_ok=True)

    csv = Path(cfg.data_root) / cfg.dataset / f"{split}.csv"
    df = pd.read_csv(csv)
    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")],
        errors="ignore",
    ).reset_index(drop=True)
    if max_structures is not None:
        df = df.iloc[:max_structures].copy()
    cifs = df["cif"].tolist()

    diag, srt = bool(cfg.diag_elems), bool(cfg.sort_eigenvalues)
    t0 = time.time()
    with mp.Pool(procs, initializer=_init_worker, initargs=(diag, srt)) as pool:
        results = pool.map(_feat_one, cifs, chunksize=200)

    keep: list[bool] = []
    rows: list[np.ndarray] = []
    nsites: list[int] = []
    failed = toobig = 0
    for r in results:
        if r is None:
            keep.append(False)
            failed += 1
        elif isinstance(r[0], str):  # ('TOOBIG', n)
            keep.append(False)
            toobig += 1
        else:
            keep.append(True)
            rows.append(r[0])
            nsites.append(r[1])

    if toobig:
        raise RuntimeError(
            f"{toobig} structure(s) exceed MAX_ATOMS={MAX_ATOMS}; "
            "raise the constant."
        )
    feats = np.vstack(rows).astype(np.float32)
    mask = np.array(keep)
    meta = df.loc[
        mask, ["material_id", "pretty_formula", "spacegroup.number"] + _PROPS
    ].copy()
    meta = meta.reset_index(drop=True)
    meta["n_sites"] = nsites
    meta["split"] = split

    np.savez_compressed(out / f"X_{split}.npz", X=feats)
    meta.to_parquet(out / f"meta_{split}.parquet", index=False)
    write_descriptor(out, descriptor_spec(diag, srt))
    log.info(
        "%s: %d structures featurized in %.1fs | X=%s | parse_fail=%d",
        split,
        len(meta),
        time.time() - t0,
        feats.shape,
        failed,
    )
    return out


def build_cache(
    cfg: PipelineConfig,
    *,
    procs: int = 4,
    splits: Sequence[str] | None = None,
    out_dir: Path | None = None,
    max_structures: int | None = None,
) -> Path:
    """Featurize the requested splits (default: all in ``cfg.splits``)."""
    cfg = cfg.resolved()
    for split in splits if splits is not None else cfg.splits:
        out = build_cache_split(
            cfg,
            split,
            procs=procs,
            out_dir=out_dir,
            max_structures=max_structures,
        )
    return out
