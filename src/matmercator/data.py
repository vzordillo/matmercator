"""Load CDVAE-style crystal datasets into pymatgen Structures + a property table.

Supported CSV datasets: mp_20, carbon_24, perov_5. Each row is one structure
with columns:

    material_id, formation_energy_per_atom [eV/atom], band_gap [eV],
    pretty_formula, e_above_hull [eV/atom], elements, cif, spacegroup.number

The ``cif`` field is a full P1 CIF string (symmetry already expanded to explicit
sites), so pymatgen parses it directly; no spglib symmetrization is needed to
recover the geometry. The integer ``spacegroup.number`` column is the dataset's
own symmetry label and is kept as-is.

(alex_mp_20 ships as parquet/json of pymatgen ComputedStructureEntry objects and
would use a different loader; it is out of scope for this CSV path.)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

CSV_DATASETS = {"mp_20", "carbon_24", "perov_5"}


def _parse_cif(cif: str) -> Structure | None:
    """Parse one CIF string to a Structure.

    Failures are surfaced as a count by the caller rather than silently imputed
    -- a dropped structure is a known, reported gap.

    Args:
        cif: A CIF string (expected to be fully expanded P1).

    Returns:
        The parsed ``Structure``, or ``None`` if parsing fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the pymatgen CIF parser is chatty
            return Structure.from_str(cif, fmt="cif")
    except Exception:
        return None


def load_split(
    data_root: str | Path,
    dataset: str,
    split: str,
    max_structures: int | None = None,
) -> pd.DataFrame:
    """Load one split into a DataFrame with a parsed ``structure`` column.

    The raw ``cif`` text column is dropped after parsing to keep memory down;
    everything downstream uses the parsed ``Structure`` objects.

    Args:
        data_root: Folder containing the ``{dataset}/{split}.csv`` files.
        dataset: Dataset name (one of ``CSV_DATASETS``).
        split: Split name, e.g. ``"train"``, ``"val"`` or ``"test"``.
        max_structures: Optional cap on the rows loaded from this split.

    Returns:
        A DataFrame with the metadata columns plus ``structure``, ``n_sites``
        and ``split``, with unparseable rows dropped.

    Raises:
        FileNotFoundError: If the split CSV does not exist.
    """
    csv = Path(data_root) / dataset / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing dataset file: {csv}")

    df = pd.read_csv(csv)
    df = df.drop(
        columns=[c for c in df.columns if c.startswith("Unnamed")],
        errors="ignore",
    )
    if max_structures is not None:
        df = df.iloc[:max_structures].copy()

    parsed = [_parse_cif(c) for c in df["cif"]]
    ok = [s is not None for s in parsed]
    n_fail = len(ok) - sum(ok)
    if n_fail:
        warnings.warn(
            f"{dataset}/{split}: dropped {n_fail} unparseable CIF(s)",
            stacklevel=2,
        )

    df = df.loc[ok].reset_index(drop=True)
    df["structure"] = [s for s in parsed if s is not None]
    df["n_sites"] = [len(s) for s in df["structure"]]
    df["split"] = split
    return df.drop(columns=["cif"])


def load_dataset(
    data_root: str | Path,
    dataset: str,
    splits: Iterable[str] = ("train", "val", "test"),
    max_structures: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate the requested splits into a single DataFrame.

    Args:
        data_root: Folder containing the ``{dataset}/{split}.csv`` files.
        dataset: Dataset name; must be one of ``CSV_DATASETS``.
        splits: Splits to load and concatenate.
        max_structures: Optional cap on the rows loaded per split.

    Returns:
        The concatenated DataFrame across ``splits``.

    Raises:
        ValueError: If ``dataset`` is not a supported CSV dataset.
    """
    if dataset not in CSV_DATASETS:
        raise ValueError(
            f"{dataset!r} is not a CSV dataset {sorted(CSV_DATASETS)}; "
            "alex_mp_20 needs a parquet/ComputedStructureEntry loader."
        )
    frames = [load_split(data_root, dataset, s, max_structures) for s in splits]
    return pd.concat(frames, ignore_index=True)
