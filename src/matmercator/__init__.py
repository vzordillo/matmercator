"""matmercator — materials-space cartography pipeline.

Adapts the Generative Topographic Mapping (GTM) "chemical cartography" paradigm
to crystal-structure data. The pipeline loads a CDVAE-style crystal dataset,
featurizes each structure into a fixed-length descriptor, fits a GTM manifold on
a stratified *frame set*, projects the full set onto the frozen map, and
color-codes the landscape by physical properties.

The descriptor is a swappable stage; the current default is the Sine Coulomb
Matrix (a cheap composition-plus-geometry descriptor). Richer descriptors (e.g.
the Orbital Field Matrix) and alternative manifolds (e.g. a Self-Organizing Map)
are planned extensions, and the module boundaries leave room to add them.
"""

from matmercator.config import PipelineConfig

__all__ = ["PipelineConfig"]
__version__ = "0.1.0"
