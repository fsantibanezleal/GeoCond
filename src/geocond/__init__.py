"""Support-aware spatial conditioning; see the versioned API contract."""

__version__ = "0.06.002"

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
from .direct_sampling import DirectSamplingResult, direct_sampling
from .geometry import Survey, TrajectorySamples, tangent
from .kriging import Observations, PredictionBatch, predict, support_covariance
from .neighborhood import Neighborhood
from .probability import IndicatorResult, bounded_isotonic, indicator_kriging, pava
from .simulation import NormalScoreTransform, SimulationResult, sequential_gaussian
from .support import Support, block_support, line_support, point_support, trajectory_support
from .validation import CancelledError, NumericalError, ValidationError
from .variogram import (
    ExperimentalVariogram,
    VariogramFit,
    experimental_cross_variogram,
    experimental_variogram,
    fit_lmc,
    fit_variogram,
)

__all__ = [
    "FAMILIES",
    "CancelledError",
    "Composite",
    "CovarianceComponent",
    "CovarianceModel",
    "DirectSamplingResult",
    "ExperimentalVariogram",
    "IndicatorResult",
    "Neighborhood",
    "NormalScoreTransform",
    "NumericalError",
    "Observations",
    "PredictionBatch",
    "SimulationResult",
    "Support",
    "Survey",
    "TrajectorySamples",
    "ValidationError",
    "VariogramFit",
    "block_support",
    "bounded_isotonic",
    "composite_categories",
    "composite_intervals",
    "correlation",
    "direct_sampling",
    "experimental_cross_variogram",
    "experimental_variogram",
    "fit_lmc",
    "fit_variogram",
    "fixed_boundaries",
    "from_gstat",
    "from_pykrige",
    "indicator_kriging",
    "inverse_distance",
    "line_support",
    "nearest_neighbour",
    "pava",
    "point_support",
    "predict",
    "principal_frame",
    "psd_matrix",
    "sequential_gaussian",
    "support_covariance",
    "tangent",
    "to_gstat",
    "to_gstools",
    "to_pykrige",
    "trajectory_support",
]
