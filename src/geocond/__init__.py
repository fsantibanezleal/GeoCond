"""Support-aware spatial conditioning; see the versioned API contract."""

__version__ = "0.02.000"

from .compositing import Composite, composite_categories, composite_intervals, fixed_boundaries
from .conventions import from_gstat, from_pykrige, to_gstat, to_gstools, to_pykrige
from .covariance import (
    FAMILIES,
    CovarianceComponent,
    CovarianceModel,
    correlation,
    principal_frame,
    psd_matrix,
)
from .geometry import Survey, TrajectorySamples, tangent
from .support import Support, block_support, line_support, point_support, trajectory_support
from .validation import CancelledError, NumericalError, ValidationError
from .variogram import (
    ExperimentalVariogram,
    VariogramFit,
    experimental_cross_variogram,
    experimental_variogram,
    fit_variogram,
)

__all__ = [
    "FAMILIES",
    "CancelledError",
    "Composite",
    "CovarianceComponent",
    "CovarianceModel",
    "ExperimentalVariogram",
    "NumericalError",
    "Support",
    "Survey",
    "TrajectorySamples",
    "ValidationError",
    "VariogramFit",
    "block_support",
    "composite_categories",
    "composite_intervals",
    "correlation",
    "experimental_cross_variogram",
    "experimental_variogram",
    "fit_variogram",
    "fixed_boundaries",
    "from_gstat",
    "from_pykrige",
    "line_support",
    "point_support",
    "principal_frame",
    "psd_matrix",
    "tangent",
    "to_gstat",
    "to_gstools",
    "to_pykrige",
    "trajectory_support",
]
