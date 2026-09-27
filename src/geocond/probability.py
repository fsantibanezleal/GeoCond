"""Multiple-indicator kriging: a discretized conditional distribution at ordered thresholds.

For thresholds t_1 < ... < t_K, each observation is coded i_k = [z <= t_k], and ordinary kriging of each indicator, with
its own covariance model and one shared neighbourhood plan, estimates F_k = P(Z <= t_k | data). The raw estimates need
not be ordered or lie in [0, 1]; the published distribution is their bounded isotonic projection, the least-squares
nearest sequence with 0 <= F_1 <= ... <= F_K <= 1. That projection is the pool-adjacent-violators (PAVA) solution
clipped to [0, 1]: clipping a nondecreasing sequence keeps it nondecreasing, and the box and the order constraints
separate for least squares. The raw estimates and the size of the correction are kept with the result.

Probabilities exist at the fitted thresholds only. Between them the distribution is interpolated linearly, which keeps
it monotone; below the first or above the last threshold it is an unmodeled tail and is reported as such, never
extrapolated, and no mean is computed from unspecified tails.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .covariance import CovarianceModel
from .kriging import Observations, predict
from .neighborhood import Neighborhood
from .support import Support
from .validation import ValidationError, array


def pava(values: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """The weighted least-squares nondecreasing fit (pool adjacent violators), written from the definition."""
    y = array(values, "values", ndim=1)
    w = np.ones(len(y)) if weights is None else array(weights, "weights", ndim=1)
    if w.shape != y.shape or np.any(w <= 0):
        raise ValidationError("weights must be positive, one per value")
    blocks: list[list[float]] = []  # [weighted mean, total weight, count]
    for yi, wi in zip(y, w, strict=True):
        blocks.append([float(yi), float(wi), 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, m) for m, _, n in blocks]) if blocks else np.zeros(0)


def bounded_isotonic(values: ArrayLike) -> NDArray[np.float64]:
    """The least-squares projection onto 0 <= F_1 <= ... <= F_K <= 1."""
    return np.clip(pava(values), 0.0, 1.0)


@dataclass(frozen=True)
class IndicatorResult:
    """Per target: the raw and corrected CDF at the thresholds, the correction's size, and the per-threshold status."""

    thresholds: NDArray[np.float64]
    raw: NDArray[np.float64]
    cdf: NDArray[np.float64]
    correction_max: NDArray[np.float64]
    correction_l2: NDArray[np.float64]
    valid: NDArray[np.bool_]
    status: tuple[str, ...]
    reasons: tuple[str | None, ...]

    def probability_below(self, value: float) -> tuple[NDArray[np.float64], str]:
        """F(value) for every target, interpolated between thresholds; outside them, NaN and 'unmodeled-tail'."""
        t = self.thresholds
        if value < t[0] or value > t[-1]:
            return np.full(len(self.cdf), np.nan), "unmodeled-tail"
        k = int(np.searchsorted(t, value, side="right")) - 1
        if k >= len(t) - 1:
            return self.cdf[:, -1].copy(), "at-threshold"
        f = (value - t[k]) / (t[k + 1] - t[k])
        return (1 - f) * self.cdf[:, k] + f * self.cdf[:, k + 1], "interpolated" if f > 0 else "at-threshold"

    def probability_above(self, value: float) -> tuple[NDArray[np.float64], str]:
        below, status = self.probability_below(value)
        return 1.0 - below, status


def indicator_kriging(
    observations: Observations,
    targets: Sequence[Support],
    thresholds: ArrayLike,
    models: Sequence[CovarianceModel],
    *,
    neighborhood: Neighborhood | None = None,
) -> IndicatorResult:
    """Ordinary indicator kriging at each threshold with its own fitted indicator covariance and one neighbourhood
    plan (selection depends on geometry only, so every threshold uses the same observations at a target)."""
    t = array(thresholds, "thresholds", ndim=1)
    if len(t) < 1 or np.any(np.diff(t) <= 0):
        raise ValidationError("thresholds must be finite and strictly increasing")
    models = list(models)
    if len(models) != len(t) or any(m.n_variables != 1 for m in models):
        raise ValidationError("one univariate indicator covariance model per threshold is required")
    if observations.n_variables != 1:
        raise ValidationError("indicator kriging codes one variable")
    targets = tuple(targets)
    raw = np.full((len(targets), len(t)), np.nan)
    status = ["estimated"] * len(targets)
    reasons: list[str | None] = [None] * len(targets)
    for k, (threshold, model) in enumerate(zip(t, models, strict=True)):
        coded = Observations(observations.supports, (observations.values <= threshold).astype(np.float64),
                             groups=observations.groups, ids=observations.ids,
                             error_variance=observations.error_variance)
        out = predict(coded, targets, model, method="ordinary", neighborhood=neighborhood)
        raw[:, k] = out.mean
        for i, (s, why) in enumerate(zip(out.status, out.reasons, strict=True)):
            if s != "estimated" and status[i] == "estimated":
                status[i] = s
                reasons[i] = f"threshold {threshold:g}: {why}"
    cdf = np.full_like(raw, np.nan)
    cmax = np.full(len(targets), np.nan)
    cl2 = np.full(len(targets), np.nan)
    for i in range(len(targets)):
        if status[i] == "estimated":
            cdf[i] = bounded_isotonic(raw[i])
            cmax[i] = float(np.max(np.abs(cdf[i] - raw[i])))
            cl2[i] = float(np.linalg.norm(cdf[i] - raw[i]))
    return IndicatorResult(t, raw, cdf, cmax, cl2, np.array([s == "estimated" for s in status]), tuple(status),
                           tuple(reasons))
