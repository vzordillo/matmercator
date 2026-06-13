"""Sine Coulomb Matrix (SCM) descriptor -- the pipeline's featurization stage.

Featurization is a swappable stage; SCM is the current, cheap default. SCM is
the periodic generalization of the Coulomb matrix. For a unit cell it builds

    M_ii = 0.5 * Z_i**2.4                            (diagonal self-terms)
    M_ij = (Z_i Z_j) * phi(r_i - r_j)   for i != j   (sine-periodic kernel)

and returns the **eigenvalues** of M, zero-padded to the largest cell seen in
``fit``. Two representations are available:

  * ``sort_eigenvalues=False`` (default): matminer's ``SineCoulombMatrix``
    eigenvalue vector, computed with the general solver ``np.linalg.eig`` and
    returned **unsorted**. The invariant object is then the eigenvalue
    *spectrum* (the multiset), not the element order; this matches the committed
    calibration baseline.
  * ``sort_eigenvalues=True``: diagonalize the (symmetric) sine matrix with the
    **symmetric solver** ``np.linalg.eigvalsh`` and return the eigenvalues
    sorted **descending** (pad zeros at the end). This yields real eigenvalues
    (no complex-cast warning) in a canonical order, so the vector is also
    *element-wise* permutation-invariant -- the textbook eigenspectrum
    representation. An MP-20 A/B showed it merely redistributes the per-property
    signal (no net gain), so it is off by default and offered as a recorded,
    auditable option (it is part of the descriptor identity stored with the
    feature cache; see ``matmercator.cache``).

This featurizer returns the *raw* eigenvalues. Standardization (z-scoring) is
deliberately NOT done here -- it belongs to the frozen map and is fit on the
frame set inside the cartographer, so the descriptor stays a pure per-structure
quantity with no dependence on the rest of the corpus.

Units: eigenvalues are in atomic-charge units; magnitudes are large
(~10^2-10^3) and span orders of magnitude across columns, which is exactly why
the downstream GTM step standardizes them.
"""

from __future__ import annotations

import numpy as np
from matminer.featurizers.structure import SineCoulombMatrix


class SCMFeaturizer:
    """sklearn-style featurizer returning Sine Coulomb Matrix eigenvalues.

    Wraps matminer's ``SineCoulombMatrix`` and runs single-threaded for
    determinism. The eigenvalue-vector length is fixed in ``fit`` to the largest
    cell seen, so every transformed vector shares one dimension.

    Attributes:
        n_features_: Eigenvalue-vector length, set after fit/transform.
    """

    def __init__(self, diag_elems: bool = True, sort_eigenvalues: bool = False):
        """Initialize the featurizer.

        Args:
            diag_elems: Include the diagonal (0.5 * Z**2.4) self-terms.
            sort_eigenvalues: If True, diagonalize the symmetric sine matrix with
                ``np.linalg.eigvalsh`` and return a descending, canonically
                sorted spectrum (element-wise permutation-invariant). If False
                (default), return matminer's unsorted ``np.linalg.eig`` vector.
        """
        self._sort_eigenvalues = sort_eigenvalues
        # When sorting we take the raw symmetric matrix (flatten=False) and
        # diagonalize it ourselves; otherwise we use matminer's flattened
        # eigenvalue vector directly.
        self._scm = SineCoulombMatrix(
            diag_elems=diag_elems, flatten=not sort_eigenvalues
        )
        self._scm.set_n_jobs(1)  # deterministic; avoids a multiprocessing fork
        self._max_atoms: int = 0  # set in fit; guarded by _fitted
        self.n_features_ = None
        self._fitted = False

    def fit(self, structures) -> SCMFeaturizer:
        """Set the eigenvalue-vector length to the max atom count.

        Fit on the FULL corpus (not just the frame set) so the padding length is
        the global maximum and every later vector has a consistent dimension.
        This is a length convention, not a learned statistic, so it introduces
        no information leakage into the manifold.

        Args:
            structures: Iterable of pymatgen ``Structure`` objects.

        Returns:
            This featurizer, for chaining.
        """
        structures = list(structures)
        self._scm.fit(structures)
        self._max_atoms = max((len(s) for s in structures), default=0)
        self._fitted = True
        return self

    def transform(self, structures, batch_size: int = 2000) -> np.ndarray:
        """Featurize structures into an ``(n, n_features)`` eigenvalue array.

        Args:
            structures: Iterable of pymatgen ``Structure`` objects.
            batch_size: Number of structures featurized per batch (unsorted
                path only).

        Returns:
            An ``(n, n_features)`` float array of eigenvalue vectors.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        structures = list(structures)
        if self._sort_eigenvalues:
            X = self._transform_sorted(structures)
        else:
            X = self._transform_raw(structures, batch_size)
        self.n_features_ = X.shape[1]
        return X

    def _transform_raw(self, structures, batch_size: int) -> np.ndarray:
        """Return matminer's unsorted flattened eigenvalues, in batches."""
        chunks = []
        for i in range(0, len(structures), batch_size):
            batch = structures[i : i + batch_size]
            vecs = self._scm.featurize_many(batch, pbar=False)
            chunks.append(np.asarray(vecs, dtype=float))
        return np.vstack(chunks) if chunks else np.empty((0, 0))

    def _transform_sorted(self, structures) -> np.ndarray:
        """Symmetric-solver spectrum, sorted descending and zero-padded."""
        rows = []
        for s in structures:
            mat = np.asarray(self._scm.featurize(s)[0], dtype=float)
            eigs = np.linalg.eigvalsh(mat)[::-1]  # real, descending
            row = np.zeros(self._max_atoms, dtype=np.float64)
            row[: len(eigs)] = eigs
            rows.append(row)
        return np.vstack(rows) if rows else np.empty((0, 0))

    def fit_transform(self, structures, batch_size: int = 2000) -> np.ndarray:
        """Fit on, then transform, the same structures.

        Args:
            structures: Iterable of pymatgen ``Structure`` objects.
            batch_size: Number of structures featurized per batch.

        Returns:
            An ``(n, n_features)`` float array of eigenvalue vectors.
        """
        return self.fit(structures).transform(structures, batch_size=batch_size)
