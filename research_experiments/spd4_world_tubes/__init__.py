"""Float64 reference implementation of native SPD(4) World Tubes."""

from .compiler import (
    ConfidenceBandOrderCertificate,
    FiberTrace,
    STAROpacityMapping,
    UVTTubesAdapter,
    affine_box_extrema,
    certify_confidence_band_order,
    pack_symmetric_3x3,
    pushforward_world_atoms,
    unpack_symmetric_3x3,
)
from .model import (
    AffineRayGauge,
    AmplitudeConvention,
    BlockCholeskySPD4,
    WorldAtomBatch,
    block_cholesky_from_covariance,
    covariance_from_block_cholesky,
)
from .retained_fiber import (
    DenseRetainedFiberRender,
    analytic_fiber_optical_depth,
    dense_retained_fiber_render,
    marginal_quadratic,
    retained_fiber_density,
)
from .hybrid_transfer import (
    HybridRetainedFiberRender,
    render_variance_certified_hybrid_metal,
)
from .retained_fiber_metal import (
    RetainedFiberMetal,
    RetainedFiberTileCertificate,
    render_retained_fiber_metal,
)

__all__ = [
    "AffineRayGauge",
    "AmplitudeConvention",
    "BlockCholeskySPD4",
    "ConfidenceBandOrderCertificate",
    "DenseRetainedFiberRender",
    "FiberTrace",
    "HybridRetainedFiberRender",
    "RetainedFiberMetal",
    "RetainedFiberTileCertificate",
    "STAROpacityMapping",
    "UVTTubesAdapter",
    "WorldAtomBatch",
    "affine_box_extrema",
    "analytic_fiber_optical_depth",
    "block_cholesky_from_covariance",
    "certify_confidence_band_order",
    "covariance_from_block_cholesky",
    "dense_retained_fiber_render",
    "marginal_quadratic",
    "pack_symmetric_3x3",
    "pushforward_world_atoms",
    "retained_fiber_density",
    "render_retained_fiber_metal",
    "render_variance_certified_hybrid_metal",
    "unpack_symmetric_3x3",
]
