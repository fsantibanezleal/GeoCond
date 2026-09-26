"""Experimental directional variograms and bounded, deterministic variogram fitting.

For a lag bin P(h), the classical (Matheron) estimator is

    gamma(h) = 1 / (2 N(h)) sum_{(i,j) in P(h)} (z_i - z_j)^2,

the Cressie-Hawkins robust estimator (Cressie and Hawkins 1980, doi:10.1007/BF01035243), with the bias correction
that includes the 0.045 / N^2 term, as GSTools 1.7.0 implements it (citing Webster and Oliver 2007,
doi:10.1002/9780470517277), is

    gamma(h) = [ (1/N(h)) sum |z_i - z_j|^(1/2) ]^4 / (2 (0.457 + 0.494 / N(h) + 0.045 / N(h)^2)),

and the cross semivariogram of two variables observed on the same supports is

    gamma_ab(h) = 1 / (2 N(h)) sum (a_i - a_j)(b_i - b_j).

Membership rules, all stated so tests can place pairs exactly on them:

- a pair is unordered (i < j) and pairs at exactly zero separation are excluded and counted as coincident;
- lag bin k holds separations edges[k] <= |h| < edges[k+1]; a pair exactly on an interior edge belongs to the upper
  bin and one exactly on the last edge to none;
- a direction d with angle tolerance t admits a pair when the line through h is within t of d, |h . d| >= |h| cos t,
  and a bandwidth b admits it when its distance from that line is at most b; both boundaries are inclusive, to a
  relative 1e-12;
- a downhole variogram pairs only observations of the same group and measures the separation in measured depth.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .covariance import FAMILIES, CovarianceComponent, CovarianceModel
from .validation import (
    ValidationError,
    array,
    cancel_if_requested,
    integer,
    positive,
    rotation_matrix,
)

ESTIMATORS = ("classical", "cressie-hawkins")
WEIGHTINGS = ("counts", "counts-over-lag2", "uniform")
_BOUNDARY = 1e-12
_CHUNK_ROWS = 256


@dataclass(frozen=True)
class ExperimentalVariogram:
    """Bins, their actual mean separation, pair counts, semivariance and the pairs that produced them."""

    edges: NDArray[np.float64]
    separation: NDArray[np.float64]
    counts: NDArray[np.int64]
    values: NDArray[np.float64]
    pair_i: NDArray[np.int64]
    pair_j: NDArray[np.int64]
    pair_bin: NDArray[np.int64]
    estimator: str
    direction: NDArray[np.float64] | None
    angle_tolerance: float | None
    bandwidth: float | None
    downhole: bool
    cross: bool
    population_pairs: int
    sampled: bool
    seed: int | None
    coincident_pairs: int
    meta: dict = field(default_factory=dict)

    @property
    def valid(self) -> NDArray[np.bool_]:
        return self.counts > 0


def _unit(direction: ArrayLike) -> NDArray[np.float64]:
    d = array(direction, "direction", ndim=1)
    if d.shape != (3,) or np.linalg.norm(d) == 0:
        raise ValidationError("direction must be a nonzero 3-vector")
    return d / np.linalg.norm(d)


def _pair_plan(groups: NDArray | None, n: int, same_group: bool):
    """Sorted order and, per sorted row, how many later rows it pairs with (all later rows, or later rows of its group)."""
    if groups is None:
        order = np.arange(n)
        ends = np.full(n, n)
    else:
        order = np.argsort(groups, kind="stable")
        sorted_groups = groups[order]
        boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
        group_end = np.r_[boundaries, n]
        ends = np.repeat(group_end, np.diff(np.r_[0, group_end])) if same_group else np.full(n, n)
    counts = ends - np.arange(n) - 1
    offsets = np.r_[0, np.cumsum(counts)]
    return order, counts, offsets


def _enumerate_pairs(n, groups, same_group, max_pairs, seed, cancel):
    """Yield (i, j) index blocks (original row numbers, i != j) covering the candidate population or a seeded uniform
    sample of it without replacement; returns the population size through the first yielded value."""
    order, _, offsets = _pair_plan(groups, n, same_group)
    population = int(offsets[-1])
    sample = None
    if max_pairs is not None and population > max_pairs:
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(population, size=max_pairs, replace=False))
    yield population, sample is not None
    for start in range(0, n, _CHUNK_ROWS):
        cancel_if_requested(cancel)
        stop = min(n, start + _CHUNK_ROWS)
        lo, hi = offsets[start], offsets[stop]
        if hi == lo:
            continue
        if sample is None:
            linear = np.arange(lo, hi)
        else:
            linear = sample[np.searchsorted(sample, lo) : np.searchsorted(sample, hi)]
            if not len(linear):
                continue
        rows = np.searchsorted(offsets, linear, side="right") - 1
        cols = rows + 1 + (linear - offsets[rows])
        yield order[rows], order[cols]


def _pairs_into_bins(coordinates, edges, *, direction, angle_tolerance, bandwidth, groups, downhole, depths,
                     max_pairs, seed, cancel):
    n = len(coordinates)
    stream = _enumerate_pairs(n, groups, downhole, max_pairs, seed, cancel)
    population, sampled = next(stream)
    cos_t = None if angle_tolerance is None else np.cos(np.radians(angle_tolerance))
    kept_i, kept_j, kept_bin, kept_sep = [], [], [], []
    coincident = 0
    for i, j in stream:
        if downhole:
            distance = np.abs(depths[j] - depths[i])
        else:
            h = coordinates[j] - coordinates[i]
            distance = np.linalg.norm(h, axis=1)
        zero = distance == 0.0
        coincident += int(zero.sum())
        keep = ~zero
        if not downhole and direction is not None:
            along = np.abs(h @ direction)
            if cos_t is not None:
                keep &= along >= distance * cos_t - _BOUNDARY * distance
            if bandwidth is not None:
                across = np.sqrt(np.maximum(distance**2 - along**2, 0.0))
                keep &= across <= bandwidth + _BOUNDARY * np.maximum(distance, 1.0)
        bins = np.searchsorted(edges, distance, side="right") - 1
        keep &= (bins >= 0) & (bins < len(edges) - 1)
        kept_i.append(i[keep])
        kept_j.append(j[keep])
        kept_bin.append(bins[keep])
        kept_sep.append(distance[keep])
    cat = lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)
    return (cat(kept_i, np.int64), cat(kept_j, np.int64), cat(kept_bin, np.int64), cat(kept_sep, np.float64),
            population, sampled, coincident)


def _validate_common(coordinates, edges, direction, angle_tolerance, bandwidth, groups, downhole, depths, n_values):
    coordinates = array(coordinates, "coordinates", ndim=2)
    if coordinates.shape[1] != 3 or len(coordinates) != n_values:
        raise ValidationError("coordinates must have shape (n, 3) matching the values")
    edges = array(edges, "edges", ndim=1)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValidationError("edges must be at least two increasing nonnegative separations")
    group_array = None
    if groups is not None:
        group_array = np.asarray(groups)
        if group_array.shape != (n_values,):
            raise ValidationError("groups must hold one label per observation")
    depth_array = None
    if downhole:
        if groups is None or depths is None:
            raise ValidationError("a downhole variogram needs groups (holes) and measured depths")
        if direction is not None or angle_tolerance is not None or bandwidth is not None:
            raise ValidationError("a downhole variogram has no spatial direction, angle tolerance or bandwidth")
        depth_array = array(depths, "depths", ndim=1)
        if depth_array.shape != (n_values,):
            raise ValidationError("depths must hold one measured depth per observation")
    unit = None if direction is None else _unit(direction)
    if unit is None and (angle_tolerance is not None or bandwidth is not None):
        raise ValidationError("an angle tolerance or a bandwidth needs a direction")
    if angle_tolerance is not None:
        angle_tolerance = positive(angle_tolerance, "angle_tolerance", zero=True)
        if angle_tolerance > 90:
            raise ValidationError("angle_tolerance must lie in [0, 90] degrees")
    if bandwidth is not None:
        bandwidth = positive(bandwidth, "bandwidth", zero=True)
    return coordinates, edges, unit, angle_tolerance, bandwidth, group_array, depth_array


def experimental_variogram(
    coordinates: ArrayLike,
    values: ArrayLike,
    edges: ArrayLike,
    *,
    direction: ArrayLike | None = None,
    angle_tolerance: float | None = None,
    bandwidth: float | None = None,
    estimator: str = "classical",
    groups: ArrayLike | None = None,
    downhole: bool = False,
    depths: ArrayLike | None = None,
    max_pairs: int | None = None,
    seed: int = 0,
    cancel: Callable[[], bool] | None = None,
) -> ExperimentalVariogram:
    """The direct semivariogram of ``values`` at ``coordinates`` (n, 3) in the lag bins ``edges``.

    ``max_pairs`` draws a seeded uniform sample of the candidate pairs without replacement when the population is
    larger; the result records the population, the sample flag and the seed. Values must be finite: unknown values are
    filtered by the caller, never read as zero.
    """
    if estimator not in ESTIMATORS:
        raise ValidationError(f"estimator must be one of {ESTIMATORS}")
    z = array(values, "values", ndim=1)
    coordinates, edges, unit, angle_tolerance, bandwidth, group_array, depth_array = _validate_common(
        coordinates, edges, direction, angle_tolerance, bandwidth, groups, downhole, depths, len(z)
    )
    if max_pairs is not None:
        max_pairs = integer(max_pairs, "max_pairs")
    i, j, b, sep, population, sampled, coincident = _pairs_into_bins(
        coordinates, edges, direction=unit, angle_tolerance=angle_tolerance, bandwidth=bandwidth,
        groups=group_array, downhole=downhole, depths=depth_array, max_pairs=max_pairs, seed=seed, cancel=cancel,
    )
    nbins = len(edges) - 1
    counts = np.bincount(b, minlength=nbins).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        separation = np.bincount(b, weights=sep, minlength=nbins) / counts
        difference = z[i] - z[j]
        if estimator == "classical":
            values_out = np.bincount(b, weights=difference**2, minlength=nbins) / (2.0 * counts)
        else:
            root = np.bincount(b, weights=np.sqrt(np.abs(difference)), minlength=nbins) / counts
            values_out = root**4 / (2.0 * (0.457 + 0.494 / counts + 0.045 / counts**2))
    empty = counts == 0
    separation[empty] = np.nan
    values_out[empty] = np.nan
    return ExperimentalVariogram(
        edges, separation, counts, values_out, i, j, b, estimator, unit, angle_tolerance, bandwidth, downhole,
        False, population, sampled, seed if sampled else None, coincident,
    )


def experimental_cross_variogram(
    coordinates: ArrayLike,
    first: ArrayLike,
    second: ArrayLike,
    edges: ArrayLike,
    *,
    direction: ArrayLike | None = None,
    angle_tolerance: float | None = None,
    bandwidth: float | None = None,
    groups: ArrayLike | None = None,
    downhole: bool = False,
    depths: ArrayLike | None = None,
    max_pairs: int | None = None,
    seed: int = 0,
    cancel: Callable[[], bool] | None = None,
) -> ExperimentalVariogram:
    """The classical cross semivariogram of two variables observed on the same supports (one coordinate per row,
    both values finite on every row). Variables on different supports are not paired by proximity: restrict them to
    common supports first."""
    a = array(first, "first", ndim=1)
    c = array(second, "second", ndim=1)
    if a.shape != c.shape:
        raise ValidationError("a cross variogram needs both variables on the same rows (common supports)")
    coordinates, edges, unit, angle_tolerance, bandwidth, group_array, depth_array = _validate_common(
        coordinates, edges, direction, angle_tolerance, bandwidth, groups, downhole, depths, len(a)
    )
    if max_pairs is not None:
        max_pairs = integer(max_pairs, "max_pairs")
    i, j, b, sep, population, sampled, coincident = _pairs_into_bins(
        coordinates, edges, direction=unit, angle_tolerance=angle_tolerance, bandwidth=bandwidth,
        groups=group_array, downhole=downhole, depths=depth_array, max_pairs=max_pairs, seed=seed, cancel=cancel,
    )
    nbins = len(edges) - 1
    counts = np.bincount(b, minlength=nbins).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        separation = np.bincount(b, weights=sep, minlength=nbins) / counts
        values_out = np.bincount(b, weights=(a[i] - a[j]) * (c[i] - c[j]), minlength=nbins) / (2.0 * counts)
    empty = counts == 0
    separation[empty] = np.nan
    values_out[empty] = np.nan
    return ExperimentalVariogram(
        edges, separation, counts, values_out, i, j, b, "classical", unit, angle_tolerance, bandwidth, downhole,
        True, population, sampled, seed if sampled else None, coincident,
    )


# --------------------------------------------------------------------------------------------------- fitting


@dataclass(frozen=True)
class VariogramFit:
    """A fitted model and the record of how it was fitted."""

    model: CovarianceModel
    objective: float
    weighting: str
    bins_used: int
    starts: tuple[dict, ...]
    best_start: int
    iterations: int
    converged: bool
    spectra: dict


def _lag_vectors(variogram: ExperimentalVariogram, isotropic: bool) -> NDArray[np.float64]:
    if variogram.direction is None:
        if not isotropic:
            raise ValidationError("an omnidirectional variogram can only fit an isotropic model")
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = variogram.direction
    return variogram.separation[:, None] * direction[None, :]


def fit_variogram(
    variograms: ExperimentalVariogram | Sequence[ExperimentalVariogram],
    families: Sequence[str],
    *,
    rotation: ArrayLike | None = None,
    isotropic: bool = False,
    nugget: bool = True,
    weighting: str = "counts",
    starts: int = 5,
    range_bounds: tuple[float, float] | None = None,
    max_iterations: int = 2000,
) -> VariogramFit:
    """Fit nested ``families`` (and a nugget unless disabled) to one or more direct experimental variograms by bounded
    weighted least squares from ``starts`` deterministic initializations.

    Every variogram contributes its nonempty bins at the model lag ``separation x direction``. An anisotropic fit (the
    default) estimates three principal ranges per component in the frame ``rotation`` and needs directional variograms
    whose directions span three dimensions; ``isotropic=True`` fits one range per component. The objective is
    sum w (gamma_hat - gamma_model)^2 over bins, with w the pair count, the pair count over the squared lag, or one,
    normalized to sum to one and divided by the largest squared semivariance. Starts spread the initial ranges over
    the observed lags; the fit with the lowest objective is returned (the first on ties) and every start is recorded.
    """
    variograms = [variograms] if isinstance(variograms, ExperimentalVariogram) else list(variograms)
    if not variograms or any(v.cross for v in variograms):
        raise ValidationError("fit_variogram fits one or more direct experimental variograms")
    families = list(families)
    if not families or any(f not in FAMILIES for f in families):
        raise ValidationError(f"families must be a nonempty list drawn from {FAMILIES}")
    if weighting not in WEIGHTINGS:
        raise ValidationError(f"weighting must be one of {WEIGHTINGS}")
    starts = integer(starts, "starts")
    frame = rotation_matrix(rotation)
    if not isotropic:
        directions = np.array([v.direction for v in variograms if v.direction is not None])
        if len(directions) != len(variograms) or np.linalg.matrix_rank(directions, tol=1e-8) < 3:
            raise ValidationError(
                "an anisotropic fit needs directional variograms spanning three dimensions; fit isotropically or "
                "add directions"
            )
    lags, targets, weights = [], [], []
    for v in variograms:
        ok = v.valid
        lags.append(_lag_vectors(v, isotropic)[ok])
        targets.append(v.values[ok])
        if weighting == "counts":
            weights.append(v.counts[ok].astype(float))
        elif weighting == "counts-over-lag2":
            weights.append(v.counts[ok] / v.separation[ok] ** 2)
        else:
            weights.append(np.ones(ok.sum()))
    lag = np.concatenate(lags)
    target = np.concatenate(targets)
    weight = np.concatenate(weights)
    if len(target) < len(families) * (1 if isotropic else 3) + len(families) + int(nugget):
        raise ValidationError("fewer nonempty bins than fitted parameters")
    weight = weight / weight.sum()
    scale = float(np.max(np.abs(target))) or 1.0
    max_sep = float(np.max(np.linalg.norm(lag, axis=1)))
    lo, hi = range_bounds if range_bounds is not None else (0.01 * max_sep, 5.0 * max_sep)
    lo, hi = positive(lo, "range lower bound"), positive(hi, "range upper bound")
    if hi <= lo:
        raise ValidationError("range bounds must be increasing")
    k = len(families)
    n_ranges = 1 if isotropic else 3

    def unpack(theta):
        at = 0
        nug = 0.0
        if nugget:
            nug = theta[0] * scale
            at = 1
        sills = theta[at : at + k] * scale
        ranges = theta[at + k :].reshape(k, n_ranges) * max_sep
        return nug, sills, ranges

    def build(theta) -> CovarianceModel:
        nug, sills, ranges = unpack(theta)
        components = tuple(
            CovarianceComponent(f, np.repeat(r, 3) if isotropic else r, [[s]], frame)
            for f, s, r in zip(families, sills, ranges, strict=True)
        )
        return CovarianceModel(components, nugget=[[nug]])

    def objective(theta):
        nug, sills, ranges = unpack(theta)
        gamma = np.full(len(target), nug)
        for f, s, r in zip(families, sills, ranges, strict=True):
            component = CovarianceComponent(f, np.repeat(r, 3) if isotropic else r, [[1.0]], frame)
            gamma = gamma + s * (1.0 - component.correlation(lag))
        return float(np.sum(weight * (target - gamma) ** 2)) / scale**2

    bounds = ([(0.0, 10.0)] if nugget else []) + [(0.0, 10.0)] * k + [(lo / max_sep, hi / max_sep)] * (k * n_ranges)
    fractions = np.linspace(0.2, 1.5, starts) if starts > 1 else np.array([0.6])
    records = []
    best = None
    for s, fraction in enumerate(fractions):
        nugget0 = [0.1 * float(np.max(target)) / scale] if nugget else []
        sill0 = [0.9 * float(np.max(target)) / scale / k] * k
        range0 = [min(max(fraction * (c + 1) / k, lo / max_sep), hi / max_sep) for c in range(k)]
        theta0 = np.array(nugget0 + sill0 + [r for r in range0 for _ in range(n_ranges)])
        result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": max_iterations, "ftol": 1e-15, "gtol": 1e-12})
        record = {"initial": theta0.tolist(), "final": result.x.tolist(), "objective": float(result.fun),
                  "iterations": int(result.nit), "converged": bool(result.success), "message": str(result.message)}
        records.append(record)
        if best is None or result.fun < records[best]["objective"] - 1e-15:
            best = s
    chosen = records[best]
    model = build(np.array(chosen["final"]))
    return VariogramFit(model, chosen["objective"], weighting, len(target), tuple(records), best,
                        chosen["iterations"], chosen["converged"], model.spectra())
