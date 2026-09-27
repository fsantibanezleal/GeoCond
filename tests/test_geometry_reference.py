"""Minimum-curvature parity with welleng, an independent current implementation.

welleng's ``MinCurve`` takes inclination from the downward vertical and azimuth from north, both in radians, and reports
local (east, north, true vertical depth) with depth positive downward. GeoCond takes dip from the horizontal, negative
downward, and reports (east, north, elevation up). The conversion is inclination = 90 degrees + dip and z = -TVD; it is
applied once here and nowhere else.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from geocond.geometry import Survey

welleng_utils = pytest.importorskip("welleng.utils")


def _welleng(md, azimuth, dip):
    curve = welleng_utils.MinCurve(np.asarray(md, float), np.deg2rad(90.0 + np.asarray(dip, float)),
                                   np.deg2rad(np.asarray(azimuth, float)))
    return curve


def _to_enu(points):
    points = np.atleast_2d(points)
    return np.column_stack([points[:, 0], points[:, 1], -points[:, 2]])


def _random_survey(rng, stations):
    md = np.r_[0.0, np.cumsum(rng.uniform(5.0, 60.0, stations - 1))]
    azimuth = np.mod(rng.uniform(0, 360) + np.cumsum(rng.normal(0, 6, stations)), 360)
    dip = np.clip(rng.uniform(-85, -40) + np.cumsum(rng.normal(0, 3, stations)), -89.5, -1)
    return md, azimuth, dip


@pytest.mark.parametrize("seed", range(12))
def test_stations_and_arc_interpolation_match_welleng(seed):
    rng = np.random.default_rng(seed)
    md, azimuth, dip = _random_survey(rng, int(rng.integers(3, 25)))
    ours = Survey([0.0, 0.0, 0.0], md, azimuth, dip)
    theirs = _welleng(md, azimuth, dip)
    scale = 1e-9 * max(1.0, md[-1])
    assert_allclose(ours.station_points, _to_enu(theirs.poss), rtol=0, atol=scale)
    assert_allclose(np.rad2deg(ours.dogleg_radians), np.rad2deg(theirs.dogleg[1:]), rtol=0, atol=1e-9)
    depths = np.sort(rng.uniform(0, md[-1], 40))
    assert_allclose(ours.at(depths).points, _to_enu(theirs.interpolate(depths)), rtol=0, atol=scale)


def test_near_straight_and_wrapping_sections_match_welleng():
    md = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
    azimuth = np.array([358.0, 359.5, 0.5, 2.0, 2.0 + 1e-9])
    dip = np.array([-60.0, -60.0, -61.0, -61.5, -61.5])
    ours = Survey([0.0, 0.0, 0.0], md, azimuth, dip)
    theirs = _welleng(md, azimuth, dip)
    assert_allclose(ours.station_points, _to_enu(theirs.poss), rtol=0, atol=1e-9 * md[-1])
    depths = np.linspace(0, md[-1], 97)
    assert_allclose(ours.at(depths).points, _to_enu(theirs.interpolate(depths)), rtol=0, atol=1e-9 * md[-1])
