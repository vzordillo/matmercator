"""Single source of truth for every choice that governs a run.

Keeping all parameters in one dataclass means a calculation is reproducible from
the config alone (parameters + convergence criteria + the pinned versions in
requirements.txt). Nothing in the pipeline reads a magic number that is not
recorded here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

# Repo root: src/matmercator/config.py -> matmercator -> src -> repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class PipelineConfig:
    """Parameters that fully determine a single cartography run.

    Grouped into data selection, descriptor, frame-set sampling, GTM
    hyperparameters, color-coding/validation, and output location. CLI drivers
    override individual fields; the resulting config is serialized to
    ``config.json`` next to every run for reproducibility.
    """

    # ---- data -------------------------------------------------------------
    # folder containing mp_20/, carbon_24/, ...
    data_root: Path = REPO_ROOT / "data"
    dataset: str = "mp_20"
    splits: Sequence[str] = ("train", "val", "test")
    frame_split: str = "train"  # frame set is drawn from this split only
    # cap rows per split (None = all); for quick tests
    max_structures: int | None = None

    # ---- descriptor: Sine Coulomb Matrix ----------------------------------
    diag_elems: bool = True  # include diagonal (0.5 Z^2.4) self-terms
    sort_eigenvalues: bool = False  # canonical (eigh) descending spectrum

    # ---- frame set: fit the manifold on a sample, project the rest --------
    frame_set_size: int = 6000
    frame_strata: Sequence[str] = ("spacegroup.number",)  # stratify by these
    standardize: bool = True  # z-score descriptors before GTM (see README)
    random_state: int = 1234

    # ---- GTM (ugtm.eGTM) --------------------------------------------------
    gtm_k: int = 16  # latent grid is k x k nodes
    gtm_m: int = 4  # RBF grid is m x m
    gtm_s: float = 0.3  # RBF width factor
    gtm_regul: float = 0.1  # weight regularization
    gtm_niter: int = 200  # EM iterations

    # ---- color-coding / validation ----------------------------------------
    color_properties: Sequence[str] = (
        "band_gap",  # eV
        "formation_energy_per_atom",  # eV/atom
        "e_above_hull",  # eV/atom
    )
    grid_bins: int = 20  # G x G binning for the smoothness metric
    n_permutations: int = 200
    knn_k: int = 15  # neighbours for the crystal-system purity metric

    # ---- output -----------------------------------------------------------
    output_dir: Path = REPO_ROOT / "results" / "mp20_scm_gtm"

    def resolved(self) -> PipelineConfig:
        """Coerce path-like fields to ``Path`` objects (e.g. after CLI parsing).

        Returns:
            This config instance, mutated in place, for chaining.
        """
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        return self

    def to_json(self, path: str | Path) -> None:
        """Serialize the config to ``path`` as indented JSON.

        Args:
            path: Destination file. Path-valued fields are stringified so the
                output is portable across machines.
        """
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["output_dir"] = str(self.output_dir)
        Path(path).write_text(json.dumps(d, indent=2, default=str))
