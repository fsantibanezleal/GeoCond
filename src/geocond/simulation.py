"""The normal-score transform and sequential Gaussian simulation.

**Normal scores.** ``NormalScoreTransform.fit`` sorts the (declustering-weighted) training values and gives each distinct
value the standard normal quantile of its weighted mid-rank plotting position,

    p_j = (W_{<j} + w_j / 2) / W,     y_j = Phi^{-1}(p_j),

where W_{<j} is the weight of the values strictly below value j, w_j the weight of value j (all its ties), and W the
total: ties share one score, so the transform is a frozen, deterministic step table. Values and scores in between are
mapped by linear interpolation of the table. Beyond the table the tail policy decides: the default ``bounded`` policy
keeps the training extrema as endpoints (a score below the lowest maps to the minimum, and so on), which bounds
extrapolation and is recorded; a declared tail adds one knot on each side, ``(score, value)``, for a stated
broader-tail sensitivity recipe. A univariate transform does not make the joint distribution Gaussian.

**Sequential Gaussian simulation.** Each realization visits the target nodes on a seeded random path. At each node,
simple kriging with mean 0 in normal-score space, on the original conditioning data and the nodes already simulated
(within the neighbourhood), gives a conditional mean and variance; the node gets mean + sd x epsilon with a recorded
standard normal innovation epsilon, and joins the conditioning set. The neighbourhood is one search over data and
nodes together, or, with ``node_neighborhood``, a two-part search as GSLIB's ``sstrat = 0`` (Deutsch and Journel
1998): the original data by ``neighborhood`` (its group limits and minimum apply to them) and the simulated nodes by
``node_neighborhood`` (its ``max_samples`` is GSLIB's ``ncnode``). The single search lets dense nodes crowd the data
out of the system: along a line of nodes far from every datum, the nodes soon fill the neighbourhood and the
realization stops seeing the data. Hard data are never altered: a node at a
conditioning location takes its value. Every realization is back-transformed to native units; ensemble statistics
(the E-type mean, quantiles, exceedance frequencies) are computed from the native realizations, never by
back-transforming a Gaussian mean. The simulation conditions on support centroids (the point-support approximation),
and it is the CPU reference: nodes on one path are simulated one after another from their current context.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import ndtri

from .covariance import CovarianceModel
from .kriging import Observations
from .neighborhood import Neighborhood
from .support import Support
from .validation import ValidationError, array, cancel_if_requested, integer


@dataclass(frozen=True)
class NormalScoreTransform:
    """A frozen table of distinct training values and their normal scores, with its tail policy."""

    values: NDArray[np.float64]
    scores: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    tail: str = "bounded"
    lower_tail: tuple[float, float] | None = None
    upper_tail: tuple[float, float] | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        values: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        lower_tail: tuple[float, float] | None = None,
        upper_tail: tuple[float, float] | None = None,
    ) -> "NormalScoreTransform":
        z = array(values, "values", ndim=1)
        w = np.ones(len(z)) if weights is None else array(weights, "weights", ndim=1)
        if w.shape != z.shape or np.any(w <= 0):
            raise ValidationError("weights must be positive, one per value")
        distinct, inverse = np.unique(z, return_inverse=True)
        if len(distinct) < 2:
            raise ValidationError("a constant population has no normal-score transform")
        weight = np.bincount(inverse, weights=w)
        below = np.r_[0.0, np.cumsum(weight)[:-1]]
        p = (below + weight / 2) / weight.sum()
        scores = ndtri(p)
        tail = "bounded"
        if lower_tail is not None or upper_tail is not None:
            tail = "declared"
            if lower_tail is not None and not (lower_tail[0] < scores[0] and lower_tail[1] <= distinct[0]):
                raise ValidationError("a lower tail knot must lie below the lowest score and at or below the minimum")
            if upper_tail is not None and not (upper_tail[0] > scores[-1] and upper_tail[1] >= distinct[-1]):
                raise ValidationError("an upper tail knot must lie above the highest score and at or above the maximum")
        return cls(distinct, scores, p, tail, lower_tail, upper_tail,
                   {"ties": "mid-rank weighted plotting position; tied values share one score", "n": len(z),
                    "weighted": weights is not None})

    def _knots(self):
        y, z = self.scores, self.values
        if self.lower_tail is not None:
            y, z = np.r_[self.lower_tail[0], y], np.r_[self.lower_tail[1], z]
        if self.upper_tail is not None:
            y, z = np.r_[y, self.upper_tail[0]], np.r_[z, self.upper_tail[1]]
        return y, z

    def forward(self, values: ArrayLike) -> NDArray[np.float64]:
        """Native values to normal scores; beyond the table, clipped to its end knots (recorded tail policy)."""
        y, z = self._knots()
        return np.interp(np.asarray(values, dtype=np.float64), z, y)

    def backward(self, scores: ArrayLike) -> NDArray[np.float64]:
        """Normal scores to native values; beyond the table, clipped to its end knots (recorded tail policy)."""
        y, z = self._knots()
        return np.interp(np.asarray(scores, dtype=np.float64), y, z)


@dataclass(frozen=True)
class SimulationResult:
    """Realizations in native units and in normal-score space, with everything needed to reproduce them."""

    native: NDArray[np.float64]
    gaussian: NDArray[np.float64]
    paths: NDArray[np.int64]
    innovations: NDArray[np.float64]
    conditional_mean: NDArray[np.float64]
    conditional_sd: NDArray[np.float64]
    hard: NDArray[np.bool_]
    seed: int
    transform: NormalScoreTransform
    neighborhood: Neighborhood
    node_neighborhood: Neighborhood | None = None

    def etype(self) -> NDArray[np.float64]:
        """The ensemble mean of the native realizations."""
        return self.native.mean(axis=0)

    def quantile(self, q: float) -> NDArray[np.float64]:
        return np.quantile(self.native, q, axis=0)

    def exceedance(self, threshold: float) -> NDArray[np.float64]:
        return (self.native > threshold).mean(axis=0)


def sequential_gaussian(
    observations: Observations,
    targets: Sequence[Support],
    model: CovarianceModel,
    transform: NormalScoreTransform,
    *,
    realizations: int = 1,
    seed: int = 0,
    neighborhood: Neighborhood | None = None,
    node_neighborhood: Neighborhood | None = None,
    cancel: Callable[[], bool] | None = None,
) -> SimulationResult:
    """Sequential Gaussian simulation of ``observations`` (native units, one variable) at ``targets``.

    ``model`` is the covariance of the normal scores (its sill should be near one). Realization r draws its path and
    innovations from ``numpy.random.default_rng([seed, r])``, so any realization can be regenerated alone. A node whose
    centroid coincides with a conditioning centroid takes the conditioning value exactly.

    Without ``node_neighborhood``, ``neighborhood`` is one search over the data and the nodes already simulated, and
    group limits are refused (simulated nodes have no group). With it, the search has two parts: ``neighborhood``
    selects the original data with their group identities, and its minimum applies to them; ``node_neighborhood``
    selects the nodes already simulated, and its minimum is not applied. A node whose data are fewer than the minimum
    is drawn from the model sill, as in the single search.
    """
    if observations.n_variables != 1 or model.n_variables != 1:
        raise ValidationError("sequential Gaussian simulation is univariate")
    realizations = integer(realizations, "realizations")
    targets = tuple(targets)
    hood = neighborhood if neighborhood is not None else Neighborhood(max_samples=len(observations) + len(targets))
    two_part = node_neighborhood is not None
    if two_part and node_neighborhood.needs_groups:
        raise ValidationError("simulated nodes have no group: node_neighborhood cannot limit or require groups")
    if not two_part and hood.needs_groups:
        raise ValidationError("the single search mixes data and simulated nodes, which have no group: declare "
                              "node_neighborhood to limit the data per group")
    data_variables = np.zeros(len(observations), np.int64)
    data_xyz = observations.centers
    data_y = transform.forward(observations.values)
    node_xyz = np.array([t.center for t in targets])
    n = len(targets)
    # hard data at nodes: exact coincidence of centroids
    lookup = {tuple(x): i for i, x in enumerate(data_xyz)}
    hard_index = np.array([lookup.get(tuple(x), -1) for x in node_xyz])
    hard = hard_index >= 0
    gaussian = np.full((realizations, n), np.nan)
    paths = np.zeros((realizations, n), np.int64)
    innovations = np.zeros((realizations, n))
    cmean = np.zeros((realizations, n))
    csd = np.zeros((realizations, n))
    for r in range(realizations):
        rng = np.random.default_rng([seed, r])
        path = rng.permutation(n)
        paths[r] = path
        sim_xyz = np.empty((n, 3))
        sim_y = np.empty(n)
        count = 0
        for node in path:
            cancel_if_requested(cancel)
            if hard[node]:
                value = data_y[hard_index[node]]
                gaussian[r, node] = value
                cmean[r, node], csd[r, node] = value, 0.0
                innovations[r, node] = 0.0
                continue
            xyz = np.concatenate([data_xyz, sim_xyz[:count]]) if count else data_xyz
            yy = np.concatenate([data_y, sim_y[:count]]) if count else data_y
            if two_part:
                idx, _, why = hood.select(data_xyz, data_variables, observations.groups, node_xyz[node], (0,))
                if count and why is None:
                    # the nodes' minimum is not applied: the selection is complete before that check
                    near, _, _ = node_neighborhood.select(sim_xyz[:count], np.zeros(count, np.int64), None,
                                                          node_xyz[node], (0,))
                    idx = np.concatenate([idx, len(data_xyz) + near])
            else:
                idx, _, why = hood.select(xyz, np.zeros(len(xyz), np.int64), None, node_xyz[node], (0,))
            eps = rng.standard_normal()
            if why is not None:
                mu, var = 0.0, float(model.total_sill[0, 0])
            else:
                C = model.between(xyz[idx], xyz[idx])
                c = model.between(xyz[idx], node_xyz[node][None, :])[:, 0]
                try:
                    factor = cho_factor(C, lower=True)
                except LinAlgError as error:
                    raise ValidationError(
                        "the conditioning covariance is singular: duplicate conditioning locations must be resolved "
                        "before simulation"
                    ) from error
                w = cho_solve(factor, c)
                mu = float(w @ yy[idx])
                var = float(model.total_sill[0, 0] - w @ c)
                if var < -1e-10 * model.total_sill[0, 0]:
                    raise ValidationError("materially negative conditional variance")
                var = max(var, 0.0)
            value = mu + np.sqrt(var) * eps
            gaussian[r, node] = value
            cmean[r, node], csd[r, node] = mu, np.sqrt(var)
            innovations[r, node] = eps
            sim_xyz[count] = node_xyz[node]
            sim_y[count] = value
            count += 1
    native = transform.backward(gaussian)
    native[:, hard] = observations.values[hard_index[hard]]
    return SimulationResult(native, gaussian, paths, innovations, cmean, csd, hard, int(seed), transform, hood,
                            node_neighborhood)
