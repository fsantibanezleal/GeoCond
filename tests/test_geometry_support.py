import numpy as np
import pytest
from numpy.testing import assert_allclose

from geocond.geometry import Survey, tangent
from geocond.support import Support, block_support, line_support, point_support, trajectory_support
from geocond.compositing import composite_categories, composite_intervals, fixed_boundaries
from geocond.validation import ValidationError


def test_straight_directions_and_arbitrary_tangent():
    for az, dip in [(0, -90), (90, 0), (359, -37)]:
        survey = Survey([12, 34, 500], [0, 100], [az, az], [dip, dip])
        depths = np.array([0, 17, 100])
        assert_allclose(
            survey.at(depths).points, survey.collar + depths[:, None] * tangent(az, dip), atol=1e-12
        )


def test_quarter_arc_and_inserted_station_are_same_curve():
    length = 100.0
    original = Survey([0, 0, 0], [0, length], [0, 0], [0, -90])
    refined = Survey([0, 0, 0], [0, 50, length], [0, 0, 0], [0, -45, -90])
    depths = np.linspace(0, length, 101)
    theta = depths / length * np.pi / 2
    exact = np.column_stack(
        [
            np.zeros_like(theta),
            2 * length / np.pi * np.sin(theta),
            -2 * length / np.pi * (1 - np.cos(theta)),
        ]
    )
    assert_allclose(original.at(depths).points, exact, atol=1e-12)
    assert_allclose(refined.at(depths).points, exact, atol=1e-12)
    assert not np.allclose(original.at([50]).points[0], original.station_points[-1] / 2)


def test_azimuth_wrap_near_zero_and_antiparallel():
    survey = Survey([0, 0, 0], [0, 100], [359, 1], [0, 0])
    assert abs(survey.station_points[-1, 0]) < 1e-12
    assert survey.station_points[-1, 1] > 99.9
    tiny = Survey([0, 0, 0], [0, 100], [1, 1 + 1e-9], [-20, -20])
    assert np.isfinite(tiny.at([50]).points).all()
    with pytest.raises(ValidationError, match="Antiparallel"):
        Survey([0, 0, 0], [0, 100], [0, 180], [0, 0])


def test_extensions_are_explicit_and_inputs_are_owned():
    with pytest.raises(ValidationError, match="First station"):
        Survey([0, 0, 0], [10, 50], [0, 0], [-90, -90])
    md = np.array([10.0, 50.0])
    survey = Survey([0, 0, 0], md, [0, 0], [-90, -90], start_extension="tangent")
    md[1] = 1000
    assert survey.measured_depth[-1] == 50
    assert survey.at([0, 10, 25]).extended.tolist() == [True, False, False]
    with pytest.raises(ValidationError, match="exceeds"):
        survey.at([51])
    with pytest.raises(ValueError):
        survey.collar[0] = 3


@pytest.mark.parametrize(
    "md,az,dip",
    [([0, 0], [0, 0], [0, 0]), ([0, 1], [0, 0], [0, -91]), ([0, 1], [0, np.inf], [0, 0])],
)
def test_bad_survey_rejected(md, az, dip):
    with pytest.raises(ValidationError):
        Survey([0, 0, 0], md, az, dip)


def test_support_quadrature_integrates_polynomials_and_arc():
    line = line_support([0, 0, 0], [2, 0, 0], order=3)
    assert_allclose(line.weights @ line.points[:, 0] ** 4, 16 / 5)
    block = block_support([1, 2, 3], [2, 4, 6], order=(3, 4, 5))
    assert_allclose(block.center, [1, 2, 3])
    assert_allclose(block.weights @ (block.points[:, 1] - 2) ** 2, 4 / 3)
    survey = Survey([0, 0, 0], [0, 50, 100], [0, 0, 0], [0, -45, -90])
    support = trajectory_support(survey, 0, 100, order=8)
    assert_allclose(support.center, [0, 400 / np.pi**2, 400 / np.pi**2 - 200 / np.pi], atol=1e-12)
    assert support.measure == "continuous"
    assert point_support([0, 0, 0]).measure == "discrete"


def test_unknown_support_and_bad_weights_not_accepted():
    for kwargs in [{"kind": "sampling-envelope"}, {"kind": "weighted"}]:
        with pytest.raises(ValidationError):
            Support([[0, 0, 0]], [1], **kwargs)
    with pytest.raises(ValidationError):
        Support([[0, 0, 0], [1, 1, 1]], [0.5, -0.5], "weighted", measure="discrete")
    with pytest.raises(ValidationError):
        block_support([0, 0, 0], [1, 1, 1], rotation=np.diag([-1, 1, 1]))


def test_composite_conservation_coverage_missing_and_domains():
    result = composite_intervals([0, 1], [1, 3], [2, 5], [0, 2, 3], source_ids=["s0", "s1"])
    assert_allclose([c.mean for c in result], [3.5, 5])
    assert sum(c.numerator for c in result) == 12
    assert sum(c.valid_length for c in result) == 3
    assert result[0].parents == (("s0", 1), ("s1", 1))
    missing = composite_intervals([0, 2, 3], [1, 3, 4], [0, np.nan, 8], [0, 4])[0]
    assert (missing.mean, missing.coverage, missing.missing_length) == (4, 0.5, 2)
    split = composite_intervals([0, 1], [1, 3], [2, 5], [0, 3], domains=["ore", "waste"])
    assert [c.domain for c in split] == ["ore", "waste"]
    assert_allclose([c.mean for c in split], [2, 5])


def test_composite_rejects_overlap_and_envelopes_residual_explicit():
    with pytest.raises(ValidationError, match="Overlapping"):
        composite_intervals([0, 0.5], [1, 2], [2, 3], [0, 2])
    with pytest.raises(ValidationError, match="known continuous"):
        composite_intervals([0], [1], [3], [0, 1], support_kind="sampling-envelope")
    assert_allclose(fixed_boundaries(0, 5, 2), [0, 2, 4, 5])
    assert_allclose(fixed_boundaries(0, 5, 2, residual="drop"), [0, 2, 4])
    assert np.isnan(composite_intervals([0], [1], [2], [0, 2], min_coverage=0.8)[0].mean)


def test_category_proportions_do_not_average_codes_or_fill_unknown():
    result = composite_categories([0, 1, 2], [1, 2, 4], ["A", None, "B"], [0, 4])[0]
    assert result["proportions"] == {"A": 1 / 3, "B": 2 / 3}
    assert result["coverage"] == 0.75
