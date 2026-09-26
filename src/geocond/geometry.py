"""Minimum-curvature trajectories in east/north/elevation-up coordinates."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import ValidationError, array


@dataclass(frozen=True)
class TrajectorySamples:
    measured_depth: NDArray[np.float64]
    points: NDArray[np.float64]
    tangents: NDArray[np.float64]
    extended: NDArray[np.bool_]


def tangent(azimuth: ArrayLike, dip: ArrayLike, *, angle_unit: str = "degree"):
    a, d = array(azimuth, "azimuth"), array(dip, "dip")
    if a.shape != d.shape:
        raise ValidationError("azimuth and dip shapes differ")
    if angle_unit not in {"degree", "radian"}:
        raise ValidationError("angle_unit must be degree or radian")
    if angle_unit == "degree":
        a, d = np.deg2rad(a), np.deg2rad(d)
    if np.any(np.abs(d) > np.pi / 2 + 1e-14):
        raise ValidationError("dip must be between -90 and 90 degrees")
    return np.stack((np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.sin(d)), axis=-1)


def _arc(t0, t1, length, fraction):
    dot = float(np.clip(t0 @ t1, -1, 1))
    beta = float(np.arctan2(np.linalg.norm(np.cross(t0, t1)), dot))
    if np.pi - beta < 1e-7:
        raise ValidationError("Antiparallel survey tangents do not determine an arc plane")
    if beta < 1e-8:
        direction = (1 - fraction) * t0 + fraction * t1
        direction /= np.linalg.norm(direction)
        displacement = length * fraction * ((1 - fraction / 2) * t0 + fraction / 2 * t1)
        return displacement, direction, beta
    normal = (t1 - dot * t0) / np.sin(beta)
    angle = fraction * beta
    displacement = length / beta * (np.sin(angle) * t0 + 2 * np.sin(angle / 2) ** 2 * normal)
    return displacement, np.cos(angle) * t0 + np.sin(angle) * normal, beta


@dataclass(frozen=True)
class Survey:
    collar: ArrayLike
    measured_depth: ArrayLike
    azimuth: ArrayLike
    dip: ArrayLike
    angle_unit: str = "degree"
    start_extension: str = "error"
    end_extension: str = "error"
    station_points: NDArray[np.float64] = field(init=False, repr=False)
    station_tangents: NDArray[np.float64] = field(init=False, repr=False)
    dogleg_radians: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        collar = array(self.collar, "collar", ndim=1)
        md = array(self.measured_depth, "measured_depth", ndim=1)
        az = array(self.azimuth, "azimuth", ndim=1)
        dip = array(self.dip, "dip", ndim=1)
        if collar.shape != (3,) or not len(md) or az.shape != md.shape or dip.shape != md.shape:
            raise ValidationError("Survey needs a 3D collar and matching nonempty station arrays")
        if md[0] < 0 or np.any(np.diff(md) <= 0):
            raise ValidationError(
                "Measured station depths must be nonnegative and strictly increasing"
            )
        for policy in (self.start_extension, self.end_extension):
            if policy not in {"error", "tangent"}:
                raise ValidationError("Extension policy must be error or tangent")
        if md[0] > 0 and self.start_extension != "tangent":
            raise ValidationError(
                "First station is below the collar; explicit tangent extension required"
            )
        ts = tangent(az, dip, angle_unit=self.angle_unit)
        xyz = np.empty((len(md), 3))
        xyz[0] = collar + md[0] * ts[0]
        doglegs = np.empty(max(0, len(md) - 1))
        for i in range(len(md) - 1):
            delta, _, doglegs[i] = _arc(ts[i], ts[i + 1], md[i + 1] - md[i], 1)
            xyz[i + 1] = xyz[i] + delta
        for name, value in [
            ("collar", collar),
            ("measured_depth", md),
            ("azimuth", az),
            ("dip", dip),
            ("station_points", xyz),
            ("station_tangents", ts),
            ("dogleg_radians", doglegs),
        ]:
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def at(self, depths: ArrayLike) -> TrajectorySamples:
        md = array(np.atleast_1d(depths), "depths", ndim=1)
        if np.any(md < 0):
            raise ValidationError("Evaluation depths must be nonnegative")
        lo, hi = self.measured_depth[[0, -1]]
        if np.any(md < lo) and self.start_extension != "tangent":
            raise ValidationError("Evaluation precedes first survey station")
        if np.any(md > hi) and self.end_extension != "tangent":
            raise ValidationError(
                "Evaluation exceeds last station; explicit tangent extension required"
            )
        xyz, direction = np.empty((len(md), 3)), np.empty((len(md), 3))
        extended = (md < lo) | (md > hi)
        for row, depth in enumerate(md):
            if depth <= lo:
                xyz[row] = self.collar + depth * self.station_tangents[0]
                direction[row] = self.station_tangents[0]
            elif depth >= hi:
                xyz[row] = self.station_points[-1] + (depth - hi) * self.station_tangents[-1]
                direction[row] = self.station_tangents[-1]
            else:
                i = np.searchsorted(self.measured_depth, depth, side="right") - 1
                length = self.measured_depth[i + 1] - self.measured_depth[i]
                delta, direction[row], _ = _arc(
                    self.station_tangents[i],
                    self.station_tangents[i + 1],
                    length,
                    (depth - self.measured_depth[i]) / length,
                )
                xyz[row] = self.station_points[i] + delta
        for item in (xyz, direction, extended):
            item.setflags(write=False)
        return TrajectorySamples(md, xyz, direction, extended)
