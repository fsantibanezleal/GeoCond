"""Deterministic local neighbourhoods on support centroids under a declared anisotropic distance.

Selection is not a covariance transformation: it only decides which observations enter a local system. Candidates are
ranked by distance from the target support's centroid, ties broken by observation order, then taken in that order
subject to the radius, the per-variable (or total) maximum and the per-group maximum. A neighbourhood that ends with
fewer observations of the target variable than ``min_samples``, or fewer distinct groups than ``min_groups``, is
uninformed: the target gets an explicit status, never an estimate from too little support.
"""

from dataclasses import dataclass
from math import inf

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import ValidationError, array, integer, positive, rotation_matrix


@dataclass(frozen=True)
class Neighborhood:
    """``metric`` is ``(rotation, ranges)``: distances are |diag(1/ranges) R^T h|, so the radius is then in range
    units (1 = one range along every principal direction). Without a metric, distances are Euclidean metres."""

    max_samples: int = 64
    min_samples: int = 1
    radius: float = inf
    max_per_group: int | None = None
    min_groups: int = 1
    per_variable: bool = True
    metric: tuple | None = None

    def __post_init__(self):
        integer(self.max_samples, "max_samples")
        integer(self.min_samples, "min_samples")
        if self.min_samples > self.max_samples:
            raise ValidationError("min_samples cannot exceed max_samples")
        if not (self.radius == inf or positive(self.radius, "radius")):
            raise ValidationError("radius must be positive or infinite")
        if self.max_per_group is not None:
            integer(self.max_per_group, "max_per_group")
        integer(self.min_groups, "min_groups")
        if self.metric is not None:
            rotation, ranges = self.metric
            ranges = array(ranges, "metric ranges", ndim=1)
            if ranges.shape != (3,) or np.any(ranges <= 0):
                raise ValidationError("metric ranges must be three positive values")
            object.__setattr__(self, "metric", (rotation_matrix(rotation), ranges))

    @property
    def needs_groups(self) -> bool:
        return self.max_per_group is not None or self.min_groups > 1

    def distances(self, centers: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
        h = centers - target[None, :]
        if self.metric is None:
            return np.linalg.norm(h, axis=1)
        rotation, ranges = self.metric
        return np.linalg.norm((h @ rotation) / ranges, axis=1)

    def select(
        self,
        centers: NDArray[np.float64],
        variables: NDArray[np.int64],
        groups: ArrayLike | None,
        target: NDArray[np.float64],
        target_variables: tuple[int, ...],
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], str | None]:
        """The selected observation indices (ascending), their distances, and the reason when uninformed."""
        if self.needs_groups and groups is None:
            raise ValidationError("group limits need the observations' group identities")
        d = self.distances(centers, target)
        order = np.lexsort((np.arange(len(d)), d))
        chosen: list[int] = []
        per_variable: dict[int, int] = {}
        per_group: dict = {}
        for i in order:
            if d[i] > self.radius:
                break
            v = int(variables[i])
            if self.per_variable:
                if per_variable.get(v, 0) >= self.max_samples:
                    continue
            elif len(chosen) >= self.max_samples:
                break
            if self.max_per_group is not None:
                g = groups[i]
                if per_group.get(g, 0) >= self.max_per_group:
                    continue
                per_group[g] = per_group.get(g, 0) + 1
            per_variable[v] = per_variable.get(v, 0) + 1
            chosen.append(int(i))
        selected = np.array(sorted(chosen), dtype=np.int64)
        for t in target_variables:
            if per_variable.get(t, 0) < self.min_samples:
                return selected, d[selected], (
                    f"{per_variable.get(t, 0)} observations of variable {t} within the neighbourhood, fewer than "
                    f"min_samples = {self.min_samples}"
                )
        if self.min_groups > 1:
            distinct = len({groups[i] for i in selected})
            if distinct < self.min_groups:
                return selected, d[selected], (
                    f"{distinct} distinct groups within the neighbourhood, fewer than min_groups = {self.min_groups}"
                )
        return selected, d[selected], None
