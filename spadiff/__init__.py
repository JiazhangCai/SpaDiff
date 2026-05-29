"""SpaDiff: Mass-particle reverse denoising for spatial transcriptomics.

Public API
----------
denoise(spot_coords, counts, ...) -> dict
    Run SpaDiff denoising on a spatial transcriptomics count matrix.

Example
-------
>>> from spadiff import denoise
>>> result = denoise(spot_coords, counts, T=15, eta_base=0.005)
>>> denoised_counts = result['counts_denoised']
"""

from .utils import denoise

__version__ = "1.0.0"
__all__ = ["denoise", "__version__"]
