"""Nearest-neighbour and inverse-distance baselines on the same eligible observations as kriging.

They are lower-complexity references a kriging result must be compared with, not estimators of uncertainty: their
variances are NaN with that stated, never a number that reads like a kriging variance. Distances are between support
centroids under the neighbourhood's declared metric; ties go to the earlier observation.
"""

from collections.abc import Sequence

import numpy as np

from .kriging import Observations, PredictionBatch
from .neighborhood import Neighborhood
from .support import Support
from .validation import ValidationError, positive

NO_VARIANCE = "a baseline estimator implies no uncertainty variance"


def _batch(means, targets, diagnostics, method, variable, status, reasons):
    m = len(targets)
    means = np.asarray(means, dtype=np.float64).reshape(m, 1)
    return PredictionBatch(means, np.full((m, 1), np.nan), np.full((m, 1, 1), np.nan),
                           np.array([s == "estimated" for s in status]), tuple(status), tuple(reasons),
                           tuple(diagnostics), method, (variable,))


def nearest_neighbour(
    observations: Observations, targets: Sequence[Support], *, variable: int = 0,
    neighborhood: Neighborhood | None = None,
) -> PredictionBatch:
    """The value of the nearest observation of ``variable`` to each target's centroid."""
    return inverse_distance(observations, targets, variable=variable, neighborhood=neighborhood, power=None)


def inverse_distance(
    observations: Observations, targets: Sequence[Support], *, variable: int = 0, power: float | None = 2.0,
    neighborhood: Neighborhood | None = None,
) -> PredictionBatch:
    """Weights proportional to distance^-power over the neighbourhood's observations of ``variable``; a target that
    coincides with observations takes their mean. ``power=None`` is the nearest neighbour."""
    if power is not None:
        power = positive(power, "power")
    keep = np.flatnonzero(observations.variables == variable)
    if not len(keep):
        raise ValidationError(f"no observations of variable {variable}")
    centers = observations.centers[keep]
    variables = observations.variables[keep]
    groups = None if observations.groups is None else observations.groups[keep]
    hood = neighborhood if neighborhood is not None else Neighborhood(max_samples=len(keep))
    means, status, reasons, diagnostics = [], [], [], []
    for target in targets:
        idx, dist, why = hood.select(centers, variables, groups, target.center, (variable,))
        if why is not None:
            means.append(np.nan)
            status.append("uninformed")
            reasons.append(why)
            diagnostics.append({"ids": observations.ids[keep[idx]].tolist()})
            continue
        z = observations.values[keep[idx]]
        coincident = dist == 0.0
        if power is None:
            nearest = int(np.lexsort((idx, dist))[0])
            weights = np.zeros(len(idx))
            weights[nearest] = 1.0
        elif coincident.any():
            weights = coincident / coincident.sum()
        else:
            weights = dist ** (-power)
            weights = weights / weights.sum()
        means.append(float(weights @ z))
        status.append("estimated")
        reasons.append(NO_VARIANCE)
        diagnostics.append({"ids": observations.ids[keep[idx]].tolist(), "distances": dist.tolist(),
                            "weights": weights.tolist()})
    return _batch(means, targets, diagnostics, "nearest" if power is None else f"idw-{power:g}", variable, status,
                  reasons)
