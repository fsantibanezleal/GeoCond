"""Support-aware spatial conditioning; see the versioned API contract."""

__version__ = "0.03.000"

from .baselines import inverse_distance, nearest_neighbour
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
from .kriging import Observations, PredictionBatch, predict, support_covariance
from .neighborhood import Neighborhood
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
    "Neighborhood",
    "NumericalError",
    "Observations",
    "PredictionBatch",
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
    "inverse_distance",
    "line_support",
    "nearest_neighbour",
    "point_support",
    "predict",
    "principal_frame",
    "psd_matrix",
    "support_covariance",
    "tangent",
    "to_gstat",
    "to_gstools",
    "to_pykrige",
    "trajectory_support",
]
