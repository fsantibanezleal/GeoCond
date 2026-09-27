"""Simple, ordinary and universal kriging and coupled cokriging on support-integrated covariance.

One local system per target support. With C the covariance of the selected observations (plus declared measurement
error on its diagonal), c the covariance of each observation with the target support, F the drift design and f0 the
drift at the target, the constrained weights solve

    [ C   F ] [ w      ]   [ c  ]
    [ F^T 0 ] [ lambda ] = [ f0 ],     z_hat = w^T z,

and the error variance is the explicit quadratic  sigma^2 = C00 - 2 w^T c + w^T C w,  which avoids any sign convention
on the Lagrange multipliers. Simple kriging has no drift and predicts m_t + w^T (z - m); ordinary kriging has one
indicator column per variable present, with the target variable's constraint equal to one and every other variable's
equal to zero (traditional ordinary cokriging); universal kriging uses a declared drift, averaged over each support and
centered and scaled before its rank is checked. A rank-deficient drift, a singular covariance, too few observations or
a materially negative variance is reported as a status with its reason and NaN values: a method is never silently
switched (universal to ordinary, full LMC to independent models) and a variance is never clamped.

Support-integrated covariance averages the model over each support's quadrature. The process nugget integrates to zero
over a continuous support (a line, a trajectory interval or a block volume) and enters between two discrete supports
only where their points coincide: a point observation keeps its nugget, a block target does not, which is gstat's
continuous-block convention. Measurement error is declared per observation, adds to the observation covariance only,
and the prediction targets the latent process.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .covariance import CovarianceModel
from .neighborhood import Neighborhood
from .support import Support
from .validation import ValidationError, array, cancel_if_requested

METHODS = ("simple", "ordinary", "universal")
#: A variance below -NEGATIVE_VARIANCE x the target's total sill is a failed solve, not a rounding residue.
NEGATIVE_VARIANCE = 1e-10
#: Drift columns whose singular values fall below this fraction of the largest are rank deficient.
DRIFT_RANK_TOLERANCE = 1e-10
#: An observation covariance whose condition number exceeds this is treated as singular: near-duplicate supports give
#: weights dominated by rounding, so the target fails unless a jitter policy is declared.
MAX_CONDITION = 1e12


def support_covariance(model: CovarianceModel, a: Support, b: Support, va: int = 0, vb: int = 0) -> float:
    """The covariance of variable ``va`` averaged over support ``a`` with variable ``vb`` averaged over ``b``."""
    h = b.points[None, :, :] - a.points[:, None, :]
    total = 0.0
    for c in model.components:
        if c.sill[va, vb] != 0.0:
            total += c.sill[va, vb] * float(a.weights @ c.correlation(h) @ b.weights)
    if model.nugget[va, vb] != 0.0 and a.measure == "discrete" and b.measure == "discrete":
        coincident = np.all(h == 0.0, axis=-1).astype(np.float64)
        total += model.nugget[va, vb] * float(a.weights @ coincident @ b.weights)
    return total


@dataclass(frozen=True)
class Observations:
    """Stacked observations, possibly of several variables at different supports. Values must be finite: unknown
    values are filtered before conditioning. Exact duplicates of one variable at one support are an error unless
    their measurement error is declared."""

    supports: Sequence[Support]
    values: ArrayLike
    variables: ArrayLike | None = None
    groups: ArrayLike | None = None
    ids: ArrayLike | None = None
    error_variance: ArrayLike | None = None

    def __post_init__(self):
        supports = tuple(self.supports)
        if not supports or not all(isinstance(s, Support) for s in supports):
            raise ValidationError("observations need one Support per value")
        n = len(supports)
        values = array(self.values, "observation values", ndim=1)
        if values.shape != (n,):
            raise ValidationError("one value per support is required")
        variables = np.zeros(n, np.int64) if self.variables is None else np.asarray(self.variables)
        if variables.shape != (n,) or variables.dtype.kind not in "iu" or np.any(variables < 0):
            raise ValidationError("variables must be one nonnegative integer index per observation")
        groups = None if self.groups is None else np.asarray(self.groups)
        if groups is not None and groups.shape != (n,):
            raise ValidationError("groups must hold one label per observation")
        ids = np.array([str(i) for i in range(n)]) if self.ids is None else np.asarray(self.ids).astype(str)
        if ids.shape != (n,) or len(set(ids.tolist())) != n:
            raise ValidationError("ids must be unique, one per observation")
        error = np.zeros(n) if self.error_variance is None else array(self.error_variance, "error_variance", ndim=1)
        if error.shape != (n,) or np.any(error < 0):
            raise ValidationError("error_variance must be one nonnegative value per observation")
        seen: dict = {}
        for i, (s, v) in enumerate(zip(supports, variables, strict=True)):
            key = (int(v), s.measure, s.points.tobytes(), s.weights.tobytes())
            if key in seen and error[i] == 0 and error[seen[key]] == 0:
                raise ValidationError(
                    f"observations {ids[seen[key]]} and {ids[i]} are the same variable at the same support; declare "
                    "their measurement error or consolidate them"
                )
            seen[key] = i
        centers = np.array([s.center for s in supports])
        for name, value in (("supports", supports), ("values", values), ("variables", variables.astype(np.int64)),
                            ("groups", groups), ("ids", ids), ("error_variance", error), ("centers", centers)):
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.supports)

    @property
    def n_variables(self) -> int:
        return int(self.variables.max()) + 1


@dataclass(frozen=True)
class PredictionBatch:
    """Per target: means and error variances for each predicted variable (columns in ``target_variables`` order),
    the joint error covariance, a status (estimated, uninformed or failed) with its reason, and diagnostics."""

    means: NDArray[np.float64]
    variances: NDArray[np.float64]
    error_covariance: NDArray[np.float64]
    valid: NDArray[np.bool_]
    status: tuple[str, ...]
    reasons: tuple[str | None, ...]
    diagnostics: tuple[dict | None, ...]
    method: str
    target_variables: tuple[int, ...]

    @property
    def mean(self) -> NDArray[np.float64]:
        """The first predicted variable's means."""
        return self.means[:, 0]

    @property
    def variance(self) -> NDArray[np.float64]:
        return self.variances[:, 0]


class _CovarianceCache:
    """Observation-to-observation covariances, reused across the targets that share neighbours."""

    def __init__(self, model: CovarianceModel, observations: Observations):
        self.model = model
        self.obs = observations
        self.cache: dict[tuple[int, int], float] = {}
        self.all_points = all(s.kind == "point" for s in observations.supports)

    def block(self, idx: NDArray[np.int64]) -> NDArray[np.float64]:
        obs, model = self.obs, self.model
        if self.all_points:
            x = obs.centers[idx]
            v = obs.variables[idx]
            out = np.empty((len(idx), len(idx)))
            for a in np.unique(v):
                for b in np.unique(v):
                    ra, rb = np.flatnonzero(v == a), np.flatnonzero(v == b)
                    out[np.ix_(ra, rb)] = model.between(x[ra], x[rb], int(a), int(b))
            return out
        out = np.empty((len(idx), len(idx)))
        for p, i in enumerate(idx):
            for q in range(p, len(idx)):
                j = idx[q]
                key = (int(i), int(j))
                if key not in self.cache:
                    self.cache[key] = support_covariance(model, obs.supports[i], obs.supports[j],
                                                         int(obs.variables[i]), int(obs.variables[j]))
                out[p, q] = out[q, p] = self.cache[key]
        return out


def _factor(C: NDArray[np.float64]):
    """The Cholesky factor of C, or None when C is not positive definite or its condition exceeds MAX_CONDITION."""
    eig = np.linalg.eigvalsh(C)
    if eig.min() <= 0 or eig.max() > MAX_CONDITION * eig.min():
        return None
    try:
        return cho_factor(C, lower=True, check_finite=True)
    except LinAlgError:
        return None


def _drift(kind, points: NDArray[np.float64], origin: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    u = (points - origin) / scale
    if kind == "linear":
        return np.c_[np.ones(len(u)), u]
    if kind == "quadratic":
        x, y, z = u.T
        return np.c_[np.ones(len(u)), u, x * x, y * y, z * z, x * y, x * z, y * z]
    if callable(kind):
        values = np.asarray(kind(points), dtype=np.float64)
        if values.ndim != 2 or len(values) != len(points) or not np.isfinite(values).all():
            raise ValidationError("a drift callable must return finite (n_points, k) covariates")
        return np.c_[np.ones(len(points)), values]
    raise ValidationError("drift must be 'linear', 'quadratic' or a callable returning covariates at points")


def _support_drift(kind, support: Support, origin, scale) -> NDArray[np.float64]:
    return support.weights @ _drift(kind, support.points, origin, scale)


def predict(
    observations: Observations,
    targets: Sequence[Support],
    model: CovarianceModel,
    *,
    method: str = "ordinary",
    target_variable: int | Sequence[int] = 0,
    mean: float | ArrayLike | None = None,
    drift: str | Callable | None = None,
    neighborhood: Neighborhood | None = None,
    solver: dict | None = None,
    backend: str = "numpy",
    cancel: Callable[[], bool] | None = None,
) -> PredictionBatch:
    """Predict ``target_variable`` (one index, or several predicted jointly) at every target support.

    ``mean`` is required by simple kriging: one value, or one per variable. ``drift`` is required by universal
    kriging (univariate only): 'linear', 'quadratic' or a callable returning covariates at points. ``solver`` may
    declare ``{"jitter": max_fraction}``: when the observation covariance is not positive definite, a jitter of
    1e-12, 1e-11, ... up to ``max_fraction`` times the largest total sill is added to its diagonal and reported;
    without it a singular covariance is a failed target.
    """
    if method not in METHODS:
        raise ValidationError(f"method must be one of {METHODS}")
    if backend != "numpy":
        raise ValidationError("backend 'numpy' is the implemented backend")
    if not isinstance(observations, Observations):
        raise ValidationError("observations must be an Observations instance")
    if observations.n_variables > model.n_variables:
        raise ValidationError("the covariance model has fewer variables than the observations")
    targets = tuple(targets)
    if not targets or not all(isinstance(t, Support) for t in targets):
        raise ValidationError("targets must be a nonempty sequence of Support")
    tv = (target_variable,) if np.isscalar(target_variable) else tuple(int(t) for t in target_variable)
    if not tv or any(t < 0 or t >= model.n_variables for t in tv):
        raise ValidationError("target variables must be indices of the covariance model")
    p = model.n_variables
    means = None
    if method == "simple":
        if mean is None:
            raise ValidationError("simple kriging needs the known mean (one value, or one per variable)")
        means = np.atleast_1d(np.asarray(mean, dtype=np.float64))
        if means.shape == (1,) and p > 1:
            raise ValidationError("simple cokriging needs one mean per variable")
        if means.shape != (p,) or not np.isfinite(means).all():
            raise ValidationError("mean must be finite, one value per model variable")
    elif mean is not None:
        raise ValidationError("a mean is used by simple kriging only")
    if method == "universal":
        if drift is None:
            raise ValidationError("universal kriging needs a declared drift")
        if p > 1 or observations.n_variables > 1:
            raise ValidationError("universal kriging is implemented for one variable")
    elif drift is not None:
        raise ValidationError("a drift is used by universal kriging only")
    jitter_max = None
    if solver is not None:
        unknown = set(solver) - {"jitter"}
        if unknown:
            raise ValidationError(f"unknown solver options {sorted(unknown)}")
        jitter_max = float(solver.get("jitter", 0.0)) or None
    hood = neighborhood if neighborhood is not None else Neighborhood(max_samples=len(observations))
    cache = _CovarianceCache(model, observations)
    obs = observations
    k = len(tv)
    m = len(targets)
    out_mean = np.full((m, k), np.nan)
    out_var = np.full((m, k), np.nan)
    out_cov = np.full((m, k, k), np.nan)
    status, reasons, diagnostics = [], [], []
    sill_scale = float(np.max(np.diag(model.total_sill)))

    for ti, target in enumerate(targets):
        cancel_if_requested(cancel)
        idx, dist, why = hood.select(obs.centers, obs.variables, obs.groups, target.center, tv)
        if why is not None:
            status.append("uninformed")
            reasons.append(why)
            diagnostics.append({"indices": idx.tolist(), "ids": obs.ids[idx].tolist()})
            continue
        z = obs.values[idx]
        v = obs.variables[idx]
        C = cache.block(idx) + np.diag(obs.error_variance[idx])
        c = np.empty((len(idx), k))
        for col, t in enumerate(tv):
            c[:, col] = [support_covariance(model, obs.supports[i], target, int(obs.variables[i]), t) for i in idx]
        C00 = np.array([[support_covariance(model, target, target, a, b) for b in tv] for a in tv])

        if method == "simple":
            F = np.zeros((len(idx), 0))
            f0 = np.zeros((0, k))
        elif method == "ordinary":
            present = np.unique(v)
            F = (v[:, None] == present[None, :]).astype(np.float64)
            missing = [t for t in tv if t not in present]
            if missing:
                status.append("uninformed")
                reasons.append(f"no observation of variable {missing[0]} in the neighbourhood")
                diagnostics.append({"indices": idx.tolist(), "ids": obs.ids[idx].tolist()})
                continue
            f0 = np.array([[1.0 if present[j] == t else 0.0 for t in tv] for j in range(len(present))])
        else:
            origin = obs.centers[idx].mean(axis=0)
            scale = float(np.max(np.abs(obs.centers[idx] - origin))) or 1.0
            F = np.array([_support_drift(drift, obs.supports[i], origin, scale) for i in idx])
            f0 = np.repeat(_support_drift(drift, target, origin, scale)[:, None], k, axis=1)
            singular = np.linalg.svd(F, compute_uv=False)
            if F.shape[0] < F.shape[1] or singular.min() < DRIFT_RANK_TOLERANCE * singular.max():
                status.append("failed")
                reasons.append("the drift is rank deficient in this neighbourhood (universal kriging is not replaced "
                               "by ordinary kriging)")
                diagnostics.append({"indices": idx.tolist(), "ids": obs.ids[idx].tolist(),
                                    "drift_singular_values": singular.tolist()})
                continue

        applied = 0.0
        factor = _factor(C)
        if factor is None and jitter_max is not None:
            fraction = 1e-12
            while factor is None and fraction <= jitter_max * (1 + 1e-9):
                factor = _factor(C + fraction * sill_scale * np.eye(len(C)))
                if factor is not None:
                    applied = fraction * sill_scale
                    C = C + applied * np.eye(len(C))
                fraction *= 10.0
        if factor is None:
            eig = np.linalg.eigvalsh(C)
            status.append("failed")
            reasons.append("the observation covariance is singular or ill conditioned (duplicate, near-duplicate or "
                           "degenerate supports); declare measurement error or a jitter policy")
            diagnostics.append({"indices": idx.tolist(), "ids": obs.ids[idx].tolist(),
                                "covariance_eigenvalue_min": float(eig.min()),
                                "covariance_eigenvalue_max": float(eig.max())})
            continue

        Ci_c = cho_solve(factor, c)
        if F.shape[1]:
            Ci_F = cho_solve(factor, F)
            S = F.T @ Ci_F
            lam = np.linalg.solve(S, F.T @ Ci_c - f0)
            W = Ci_c - Ci_F @ lam
        else:
            lam = np.zeros((0, k))
            W = Ci_c

        if method == "simple":
            prediction = means[list(tv)] + W.T @ (z - means[v])
        else:
            prediction = W.T @ z
        error_cov = C00 - W.T @ c - c.T @ W + W.T @ C @ W
        variance = np.diag(error_cov).copy()
        target_sill = np.diag(C00)
        if np.any(variance < -NEGATIVE_VARIANCE * np.maximum(target_sill, sill_scale)):
            status.append("failed")
            reasons.append("materially negative error variance: the covariance model or the solve is invalid here")
            diagnostics.append({"indices": idx.tolist(), "ids": obs.ids[idx].tolist(), "variance": variance.tolist()})
            continue

        A = np.block([[C, F], [F.T, np.zeros((F.shape[1], F.shape[1]))]])
        x = np.vstack([W, lam])
        rhs = np.vstack([c, f0])
        linear_residual = float(np.linalg.norm(A @ x - rhs) / (np.linalg.norm(A) * np.linalg.norm(x) + np.linalg.norm(rhs)))
        constraint_residual = float(np.max(np.abs(F.T @ W - f0))) if F.shape[1] else 0.0
        eig = np.linalg.eigvalsh(C)
        out_mean[ti] = prediction
        out_var[ti] = variance
        out_cov[ti] = error_cov
        status.append("estimated")
        reasons.append(None)
        weight_sums = {int(u): W[v == u].sum(axis=0).tolist() for u in np.unique(v)}
        contributions = {int(u): (W[v == u].T @ z[v == u]).tolist() for u in np.unique(v)}
        diagnostics.append({
            "indices": idx.tolist(),
            "ids": obs.ids[idx].tolist(),
            "distances": dist.tolist(),
            "weights": W.tolist(),
            "weight_sums": weight_sums,
            "contributions": contributions,
            "groups": None if obs.groups is None else len({obs.groups[i] for i in idx}),
            "lagrange": lam.tolist(),
            "linear_residual": linear_residual,
            "constraint_residual": constraint_residual,
            "covariance_eigenvalue_min": float(eig.min()),
            "covariance_eigenvalue_max": float(eig.max()),
            "condition": float(eig.max() / eig.min()) if eig.min() > 0 else float("inf"),
            "negative_weight_mass": float(np.abs(W[W < 0]).sum()),
            "regularization": applied,
        })

    valid = np.array([s == "estimated" for s in status])
    return PredictionBatch(out_mean, out_var, out_cov, valid, tuple(status), tuple(reasons), tuple(diagnostics),
                           method, tv)
