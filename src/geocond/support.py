"""Explicit known sampling measures and Gauss-Legendre support quadrature."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike

from .geometry import Survey
from .validation import ValidationError, array, integer, rotation_matrix


@dataclass(frozen=True)
class Support:
    points: ArrayLike
    weights: ArrayLike
    kind: str
    id: str = ""
    measure: str | None = None

    def __post_init__(self):
        points, weights = (
            array(self.points, "support points", ndim=2),
            array(self.weights, "support weights", ndim=1),
        )
        if points.shape != (len(weights), 3) or not len(weights):
            raise ValidationError("Support requires nonempty (n,3) points and n weights")
        if self.kind not in {"point", "line", "trajectory", "block", "weighted"}:
            raise ValidationError(
                "Support must declare a known point, line, trajectory, block or weighted measure"
            )
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
            raise ValidationError("Known support weights must be nonnegative and sum to one")
        if self.kind == "point" and len(weights) != 1:
            raise ValidationError("Point support contains exactly one coordinate")
        measure = self.measure
        if measure is None:
            if self.kind == "weighted":
                raise ValidationError(
                    "Weighted support must explicitly declare discrete or continuous measure"
                )
            measure = "discrete" if self.kind == "point" else "continuous"
        if measure not in {"continuous", "discrete"}:
            raise ValidationError("Support measure must be continuous or discrete")
        if self.kind == "point" and measure != "discrete":
            raise ValidationError("A point support is a discrete sampling measure")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "measure", measure)

    @property
    def center(self):
        return self.weights @ self.points


def point_support(point: ArrayLike, *, id: str = "") -> Support:
    point = array(point, "point", ndim=1)
    return Support(point[None, :], [1.0], "point", id)


def line_support(start: ArrayLike, end: ArrayLike, *, order: int = 8, id: str = "") -> Support:
    a, b = array(start, "start", ndim=1), array(end, "end", ndim=1)
    if a.shape != (3,) or b.shape != (3,) or np.linalg.norm(b - a) == 0:
        raise ValidationError("Line support needs distinct 3D endpoints")
    nodes, weights = leggauss(integer(order, "quadrature order"))
    return Support(a + (nodes[:, None] + 1) / 2 * (b - a), weights / 2, "line", id)


def trajectory_support(
    survey: Survey, start: float, end: float, *, order: int = 8, id: str = ""
) -> Support:
    if not np.isfinite([start, end]).all() or start < 0 or end <= start:
        raise ValidationError("Trajectory support requires a positive measured-depth interval")
    nodes, weights = leggauss(integer(order, "quadrature order"))
    boundaries = np.r_[
        start,
        survey.measured_depth[(survey.measured_depth > start) & (survey.measured_depth < end)],
        end,
    ]
    points, measure = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:], strict=True):
        points.append(survey.at(a + (nodes + 1) / 2 * (b - a)).points)
        measure.append(weights / 2 * (b - a) / (end - start))
    return Support(np.concatenate(points), np.concatenate(measure), "trajectory", id)


def block_support(
    center: ArrayLike,
    size: ArrayLike,
    *,
    order: int | tuple = 4,
    rotation: ArrayLike | None = None,
    id: str = "",
) -> Support:
    center, size = array(center, "block center", ndim=1), array(size, "block size", ndim=1)
    if center.shape != (3,) or size.shape != (3,) or np.any(size <= 0):
        raise ValidationError("Block center and positive size must each have three components")
    orders = (order,) * 3 if np.isscalar(order) else tuple(order)
    if len(orders) != 3:
        raise ValidationError("Block quadrature needs three orders")
    rules = [leggauss(integer(o, "quadrature order")) for o in orders]
    points, weights = [], []
    for indices in product(*(range(o) for o in orders)):
        points.append([rules[d][0][i] * size[d] / 2 for d, i in enumerate(indices)])
        weights.append(np.prod([rules[d][1][i] / 2 for d, i in enumerate(indices)]))
    return Support(center + np.asarray(points) @ rotation_matrix(rotation).T, weights, "block", id)
