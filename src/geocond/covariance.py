"""Nested stationary covariance models: families, geometric anisotropy and the linear model of coregionalization.

A component evaluates its correlation on the transformed separation

    r = | diag(1/a) R^T h |,

where ``a`` holds the practical principal ranges and the columns of ``R`` are the principal directions. The families
use the practical-range convention of the API contract:

    exponential  rho(r) = exp(-3 r)
    spherical    rho(r) = 1 - 1.5 r + 0.5 r^3 for r < 1, else 0
    gaussian     rho(r) = exp(-3 r^2)

A model over p variables is ``C_ab(h) = N_ab [h = 0] + sum_k (B_k)_ab rho_k(h)``: every component shares one
correlation structure and anisotropy across variables, and every sill matrix ``B_k`` and the nugget ``N`` is symmetric
positive semidefinite, which makes every assembled covariance matrix positive semidefinite. The nugget is the process
nugget (a microscale structure present at exactly zero separation); observation error is not part of a covariance
model and is declared with the observations.
"""

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import ValidationError, array, rotation_matrix

FAMILIES = ("exponential", "spherical", "gaussian")
#: Eigenvalues below -PSD_TOLERANCE x max(1, largest |eigenvalue|) reject a sill matrix as not PSD.
PSD_TOLERANCE = 1e-10


def correlation(family: str, r: ArrayLike) -> NDArray[np.float64]:
    """The correlation of ``family`` at transformed separation ``r`` (practical range at r = 1)."""
    r = np.asarray(r, dtype=np.float64)
    if family == "exponential":
        return np.exp(-3.0 * r)
    if family == "spherical":
        return np.where(r < 1.0, 1.0 - 1.5 * r + 0.5 * r**3, 0.0)
    if family == "gaussian":
        return np.exp(-3.0 * r * r)
    raise ValidationError(f"Unknown covariance family {family!r}; expected one of {FAMILIES}")


def psd_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """A symmetric positive semidefinite matrix; a scalar becomes (1, 1). Pairwise checks are not enough for three or
    more variables, so the full spectrum is tested."""
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 0:
        raw = raw.reshape(1, 1)
    matrix = array(raw, name, ndim=2)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValidationError(f"{name} must be a nonempty square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValidationError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(1.0, float(np.abs(eigenvalues).max()))
    if eigenvalues.min() < -PSD_TOLERANCE * scale:
        raise ValidationError(
            f"{name} is not positive semidefinite (smallest eigenvalue {eigenvalues.min():.3e})"
        )
    result = 0.5 * (matrix + matrix.T)
    result.setflags(write=False)
    return result


def principal_frame(
    azimuth: float, dip: float, rake: float = 0.0, *, angle_unit: str = "degree"
) -> NDArray[np.float64]:
    """The rotation whose columns are the major, semi-major and minor principal directions.

    The major direction has ``azimuth`` clockwise from north and ``dip`` negative downward, the conventions of
    ``Survey``. The semi-major direction starts horizontal, 90 degrees clockwise of the major azimuth, and both it and
    the minor direction turn about the major one by ``rake``. The frame is right handed (minor = major x semi-major).
    """
    if angle_unit not in {"degree", "radian"}:
        raise ValidationError("angle_unit must be 'degree' or 'radian'")
    az, dp, rk = (radians(v) if angle_unit == "degree" else float(v) for v in (azimuth, dip, rake))
    major = np.array([sin(az) * cos(dp), cos(az) * cos(dp), sin(dp)])
    semi = np.array([cos(az), -sin(az), 0.0])
    minor = np.cross(major, semi)
    semi, minor = cos(rk) * semi + sin(rk) * minor, -sin(rk) * semi + cos(rk) * minor
    return rotation_matrix(np.column_stack([major, semi, minor]))


@dataclass(frozen=True)
class CovarianceComponent:
    """One nested structure: a correlation family, its practical principal ranges, a principal frame and a (p, p)
    sill matrix shared by every pair of variables."""

    family: str
    ranges: ArrayLike
    sill: ArrayLike
    rotation: ArrayLike | None = None

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValidationError(f"Unknown covariance family {self.family!r}; expected one of {FAMILIES}")
        ranges = np.asarray(self.ranges, dtype=np.float64)
        if ranges.ndim == 0:
            ranges = np.full(3, float(ranges))
        ranges = array(ranges, "ranges", ndim=1)
        if ranges.shape != (3,) or np.any(ranges <= 0):
            raise ValidationError("ranges must be one positive value or three positive principal ranges")
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "sill", psd_matrix(self.sill, f"{self.family} sill"))
        object.__setattr__(self, "rotation", rotation_matrix(self.rotation))

    @property
    def n_variables(self) -> int:
        return self.sill.shape[0]

    @property
    def isotropic(self) -> bool:
        return bool(np.all(self.ranges == self.ranges[0]))

    def transformed_distance(self, h: ArrayLike) -> NDArray[np.float64]:
        """r = |diag(1/a) R^T h| for separations ``h`` of shape (..., 3)."""
        h = np.asarray(h, dtype=np.float64)
        if h.shape[-1] != 3:
            raise ValidationError("separations must have a last dimension of 3")
        return np.linalg.norm((h @ self.rotation) / self.ranges, axis=-1)

    def correlation(self, h: ArrayLike) -> NDArray[np.float64]:
        return correlation(self.family, self.transformed_distance(h))


@dataclass(frozen=True)
class CovarianceModel:
    """A nested covariance over p variables: a process nugget and components sharing their structures."""

    components: tuple[CovarianceComponent, ...]
    nugget: ArrayLike | None = None

    def __post_init__(self):
        components = tuple(self.components)
        if not all(isinstance(c, CovarianceComponent) for c in components):
            raise ValidationError("components must be CovarianceComponent instances")
        sizes = {c.n_variables for c in components}
        if self.nugget is not None:
            nugget = psd_matrix(self.nugget, "nugget")
            sizes.add(nugget.shape[0])
        if not sizes:
            raise ValidationError("A covariance model needs at least one component or a nugget")
        if len(sizes) != 1:
            raise ValidationError("Every component and the nugget must cover the same variables")
        p = sizes.pop()
        if self.nugget is None:
            nugget = np.zeros((p, p))
            nugget.setflags(write=False)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "nugget", nugget)

    @property
    def n_variables(self) -> int:
        return self.nugget.shape[0]

    @property
    def total_sill(self) -> NDArray[np.float64]:
        """C(0): the nugget plus every component's sill matrix."""
        return self.nugget + sum((c.sill for c in self.components), np.zeros_like(self.nugget))

    def _check_variables(self, a: int, b: int) -> None:
        p = self.n_variables
        if not (0 <= a < p and 0 <= b < p):
            raise ValidationError(f"variables must lie in [0, {p})")

    def covariance(self, h: ArrayLike, a: int = 0, b: int = 0) -> NDArray[np.float64]:
        """C_ab at separations ``h`` (..., 3); the nugget enters only at exactly zero separation."""
        self._check_variables(a, b)
        h = np.asarray(h, dtype=np.float64)
        result = np.where(np.all(h == 0.0, axis=-1), self.nugget[a, b], 0.0)
        for c in self.components:
            if c.sill[a, b] != 0.0:
                result = result + c.sill[a, b] * c.correlation(h)
        return result

    def variogram(self, h: ArrayLike, a: int = 0, b: int = 0) -> NDArray[np.float64]:
        """gamma_ab(h) = C_ab(0) - C_ab(h), the direct (a = b) or cross semivariogram of the model."""
        return self.total_sill[a, b] - self.covariance(h, a, b)

    def between(self, x: ArrayLike, y: ArrayLike, a: int = 0, b: int = 0) -> NDArray[np.float64]:
        """The (n, m) covariance between point sets ``x`` (n, 3) and ``y`` (m, 3), separation y - x."""
        x = array(x, "points", ndim=2)
        y = array(y, "points", ndim=2)
        if x.shape[1] != 3 or y.shape[1] != 3:
            raise ValidationError("points must have shape (n, 3)")
        return self.covariance(y[None, :, :] - x[:, None, :], a, b)

    def spectra(self) -> dict[str, NDArray[np.float64]]:
        """The eigenvalues of the nugget and of every component's sill matrix, for the fit records."""
        out = {"nugget": np.linalg.eigvalsh(self.nugget)}
        for k, c in enumerate(self.components):
            out[f"{k}:{c.family}"] = np.linalg.eigvalsh(c.sill)
        return out

    def variable(self, a: int) -> "CovarianceModel":
        """The direct model of one variable, the univariate model a zero-cross LMC must reduce to."""
        self._check_variables(a, a)
        return CovarianceModel(
            tuple(CovarianceComponent(c.family, c.ranges, [[c.sill[a, a]]], c.rotation) for c in self.components),
            nugget=[[self.nugget[a, a]]],
        )
