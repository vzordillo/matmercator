"""GTM cartography: fit a manifold on a frame set, project everything else.

Generative Topographic Mapping (GTM) is a probabilistic, generative counterpart
to the Self-Organizing Map. It fits a constrained Gaussian mixture whose centres
lie on a smooth 2-D manifold embedded in descriptor space; each object then has
a posterior ("responsibility") distribution over a k x k latent grid. We use
``model='means'``, so projecting an object returns its responsibility-weighted
mean position in the 2-D latent square -- a continuous coordinate suitable for a
scatter map.

Frame-set discipline (the scaling strategy in the feasibility study):
  * The StandardScaler and the GTM are BOTH fit only on the frame set, then
    frozen.
  * ``project`` applies the frozen scaler + manifold to any structures.

GTM training cost grows with samples x latent-nodes x dims per EM iteration, so
fitting on a representative frame set and projecting the remainder keeps a large
corpus tractable and the map stable/reproducible.

This wraps ``ugtm.eGTM`` (scikit-learn-compatible: fit / transform).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from ugtm import eGTM


def _extract_node_coords(gtm) -> np.ndarray:
    """Locate the ``(K, 2)`` latent node coordinates on a fitted eGTM.

    Args:
        gtm: A fitted ``ugtm.eGTM`` instance.

    Returns:
        The ``(K, 2)`` latent grid coordinates.

    Raises:
        AttributeError: If no node-coordinate matrix can be found.
    """
    for attr in ("optimizedModel", "optimized_model", "model"):
        obj = getattr(gtm, attr, None)
        if obj is not None and hasattr(obj, "matX"):
            return np.asarray(obj.matX, dtype=float)
    raise AttributeError(
        "could not find latent node coordinates (matX) on fitted eGTM"
    )


class GTMCartographer:
    """Fit a GTM on a frame set and project the full corpus onto it.

    The optional ``StandardScaler`` and the GTM are fit only on the frame set
    and then frozen; ``responsibilities``/``project`` apply the frozen models to
    any input. The GTM is fit at the responsibility level, and the 2-D mean
    embedding is recovered as ``R @ node_coords`` -- this loses nothing and also
    enables node-based property landscapes.
    """

    def __init__(
        self,
        k: int = 16,
        m: int = 4,
        s: float = 0.3,
        regul: float = 0.1,
        niter: int = 200,
        random_state: int = 1234,
        standardize: bool = True,
    ):
        """Configure the GTM hyperparameters.

        Args:
            k: Latent grid is ``k x k`` nodes.
            m: RBF grid is ``m x m``.
            s: RBF width factor.
            regul: Weight regularization.
            niter: EM iterations.
            random_state: Seed for the GTM.
            standardize: Z-score the descriptor (scaler fit on the frame set).
        """
        # eGTM signature (verified against ugtm 2.3.0):
        #   eGTM(k, m, s, regul, random_state, niter, verbose, model)
        self.params = dict(
            k=k, m=m, s=s, regul=regul, niter=niter, random_state=random_state
        )
        self.standardize = standardize
        self._scaler: Any = StandardScaler() if standardize else None
        self._gtm: Any = None
        self._node_coords: Any = None  # (K, 2) latent grid coordinates

    def fit(self, X_frame) -> GTMCartographer:
        """Fit the scaler and GTM on the frame set, then freeze both.

        Args:
            X_frame: ``(n_frame, d)`` descriptor matrix for the frame set.

        Returns:
            This cartographer, for chaining.
        """
        X = np.asarray(X_frame, dtype=float)
        if self.standardize:
            X = self._scaler.fit_transform(X)  # frame-set statistics, frozen
        # Fit at the responsibility level (the primary GTM object). The (n, 2)
        # mean embedding is just R @ node_coords, so storing responsibilities
        # loses nothing and also enables node-based property landscapes.
        self._gtm = eGTM(model="responsibilities", **self.params).fit(X)
        self._node_coords = _extract_node_coords(self._gtm)
        return self

    def responsibilities(self, X) -> np.ndarray:
        """Posterior responsibilities of each row of ``X`` over the K nodes.

        Args:
            X: ``(n, d)`` descriptor matrix.

        Returns:
            An ``(n, K)`` matrix whose rows sum to 1 -- the full fuzzy
            node-membership used to build property landscapes.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._gtm is None:
            raise RuntimeError("call fit() before responsibilities()")
        X = np.asarray(X, dtype=float)
        if self.standardize:
            X = self._scaler.transform(X)
        return np.asarray(self._gtm.transform(X), dtype=float)

    def project(self, X) -> np.ndarray:
        """Responsibility-weighted mean latent coordinate.

        Args:
            X: ``(n, d)`` descriptor matrix.

        Returns:
            An ``(n, 2)`` array of latent coordinates (identical to
            ``eGTM(model='means')``).
        """
        R = self.responsibilities(X)
        return R @ self._node_coords  # identical to eGTM(model='means')

    @property
    def node_coords(self) -> np.ndarray:
        """The ``(K, 2)`` latent node coordinates of the fitted manifold."""
        if self._node_coords is None:
            raise RuntimeError("call fit() first")
        return self._node_coords

    @property
    def n_nodes(self) -> int:
        """Number of latent nodes ``K`` (0 before ``fit``)."""
        return 0 if self._node_coords is None else self._node_coords.shape[0]

    def fit_project(self, X_frame, X_all) -> np.ndarray:
        """Fit on the frame set and return the projection of ``X_all``.

        Args:
            X_frame: ``(n_frame, d)`` frame-set descriptor matrix.
            X_all: ``(n, d)`` descriptor matrix to project.

        Returns:
            An ``(n, 2)`` array of latent coordinates.
        """
        return self.fit(X_frame).project(X_all)
