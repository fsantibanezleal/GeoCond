"""Covariance families, anisotropy, the LMC's positive semidefiniteness, nugget semantics and gstat parity."""

import json
from pathlib import Path

import numpy as np
import pytest

from geocond.covariance import (
    CovarianceComponent,
    CovarianceModel,
    correlation,
    principal_frame,
    psd_matrix,
)
from geocond.validation import ValidationError

REFERENCE = json.loads((Path(__file__).parent / "reference" / "gstat_fixed_covariance.json").read_text("utf-8"))


def gstat_lmc() -> CovarianceModel:
    """The reference LMC: gstat Exp(a = 4) is GeoCond's exponential with practical range 12; Sph(9) has range 9."""
    f = REFERENCE["fixture"]
    exp_a, sph_range = f["ranges"]
    return CovarianceModel(
        (
            CovarianceComponent("exponential", 3.0 * exp_a, f["components"]["exponential"]),
            CovarianceComponent("spherical", sph_range, f["components"]["spherical"]),
        ),
        nugget=f["components"]["nugget"],
    )


def test_families_follow_the_practical_range_convention():
    r = np.array([0.0, 0.5, 1.0, 1.5])
    assert np.allclose(correlation("exponential", r), np.exp(-3 * r))
    assert np.allclose(correlation("gaussian", r), np.exp(-3 * r**2))
    assert np.allclose(correlation("spherical", r), [1.0, 1 - 0.75 + 0.0625, 0.0, 0.0])
    # the spherical correlation is continuous at its range and zero beyond it
    assert correlation("spherical", 1 - 1e-9) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValidationError):
        correlation("hole-effect", r)


def test_the_full_spectrum_decides_positive_semidefiniteness():
    # pairwise-valid correlations whose 3 by 3 matrix is not PSD: pairwise checks alone would accept it
    bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert bad[a, a] * bad[b, b] >= bad[a, b] ** 2
    with pytest.raises(ValidationError, match="not positive semidefinite"):
        psd_matrix(bad, "sill")
    with pytest.raises(ValidationError, match="symmetric"):
        psd_matrix([[1.0, 0.2], [0.1, 1.0]], "sill")
    assert psd_matrix(2.5, "sill").shape == (1, 1)
    with pytest.raises(ValidationError):
        CovarianceModel((CovarianceComponent("spherical", 1, bad),))


def test_principal_frame_is_proper_and_points_where_it_says():
    frame = principal_frame(90, 0)
    assert np.allclose(frame.T @ frame, np.eye(3)) and np.isclose(np.linalg.det(frame), 1)
    assert np.allclose(frame[:, 0], [1, 0, 0])  # azimuth 90 is east
    down = principal_frame(0, -90)
    assert np.allclose(down[:, 0], [0, 0, -1], atol=1e-12)  # dip -90 is straight down
    raked = principal_frame(30, -40, 25)
    assert np.allclose(raked[:, 0], principal_frame(30, -40)[:, 0])  # rake turns about the major axis only
    assert np.isclose(np.linalg.det(raked), 1)


def test_anisotropic_ranges_act_along_their_principal_directions():
    frame = principal_frame(30, -20, 10)
    c = CovarianceComponent("exponential", [60.0, 20.0, 5.0], [[1.0]], frame)
    for axis, a in enumerate([60.0, 20.0, 5.0]):
        h = frame[:, axis] * a
        assert c.transformed_distance(h) == pytest.approx(1.0)
        assert c.correlation(h) == pytest.approx(np.exp(-3))


def test_rotating_space_and_the_frame_together_changes_nothing():
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    frame = principal_frame(40, -30, 15)
    model = CovarianceModel((CovarianceComponent("spherical", [50.0, 30.0, 10.0], [[2.0]], frame),), nugget=[[0.3]])
    turned = CovarianceModel((CovarianceComponent("spherical", [50.0, 30.0, 10.0], [[2.0]], q @ frame),), nugget=[[0.3]])
    h = rng.normal(scale=20, size=(200, 3))
    assert np.allclose(model.covariance(h), turned.covariance(h @ q.T), atol=1e-13)


def test_the_process_nugget_enters_only_at_exactly_zero_separation():
    model = CovarianceModel((), nugget=[[0.1]])
    lags = np.array([[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert model.covariance(lags).tolist() == REFERENCE["nuggetCovarianceAtZeroAndPositiveLags"]


def test_evaluated_lmc_covariances_match_gstat():
    """The reference compares evaluated covariance before any estimator, at the seven lags gstat evaluated."""
    model = gstat_lmc()
    for key, (a, b) in {"1:1": (0, 0), "1:2": (0, 1), "2:2": (1, 1)}.items():
        ref = REFERENCE["covariance"][key]
        h = np.array(ref["lags"], dtype=float)[:, None] * np.array([1.0, 0.0, 0.0])
        assert np.allclose(model.covariance(h, a, b), ref["gstat"], rtol=0, atol=2e-10), key
        assert np.allclose(model.covariance(h, a, b), model.covariance(h, b, a))


def test_component_eigenvalues_match_the_recorded_spectra():
    spectra = gstat_lmc().spectra()
    recorded = REFERENCE["fixture"]["componentEigenvalues"]
    assert np.allclose(spectra["nugget"], sorted(recorded["nugget"]))
    assert np.allclose(spectra["0:exponential"], sorted(recorded["exponential"]))
    assert np.allclose(spectra["1:spherical"], sorted(recorded["spherical"]))


def test_an_assembled_lmc_covariance_is_positive_semidefinite():
    rng = np.random.default_rng(11)
    b1 = rng.normal(size=(3, 3))
    b2 = rng.normal(size=(3, 3))
    model = CovarianceModel(
        (
            CovarianceComponent("exponential", [40.0, 25.0, 8.0], b1 @ b1.T, principal_frame(20, -10)),
            CovarianceComponent("spherical", [15.0, 15.0, 15.0], b2 @ b2.T),
        ),
        nugget=0.05 * np.eye(3),
    )
    x = rng.uniform(0, 60, size=(40, 3))
    blocks = [[model.between(x, x, a, b) for b in range(3)] for a in range(3)]
    full = np.block(blocks)
    assert np.allclose(full, full.T, atol=1e-12)
    assert np.linalg.eigvalsh(full).min() > -1e-10


def test_one_variable_of_an_lmc_is_its_direct_model():
    model = gstat_lmc()
    h = np.array([[2.0, 1.0, -0.5], [7.0, 0.0, 0.0]])
    assert np.allclose(model.variable(1).covariance(h), model.covariance(h, 1, 1))
    assert model.variable(0).n_variables == 1


def test_variogram_is_total_sill_minus_covariance():
    model = gstat_lmc()
    h = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    assert model.variogram(h, 0, 0)[0] == 0.0
    assert model.variogram(h, 0, 1)[1] == pytest.approx(model.total_sill[0, 1] - model.covariance(h, 0, 1)[1])


def test_components_must_share_their_variables():
    with pytest.raises(ValidationError, match="same variables"):
        CovarianceModel((CovarianceComponent("spherical", 1, [[1.0]]), CovarianceComponent("spherical", 1, np.eye(2))))
    with pytest.raises(ValidationError):
        CovarianceComponent("spherical", [1.0, -1.0, 1.0], [[1.0]])
