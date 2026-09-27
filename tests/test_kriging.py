"""Kriging and cokriging against R/gstat, PyKrige and GSTools, and the acceptance cases of the methods dossier."""

import json
from pathlib import Path

import numpy as np
import pytest

from geocond.baselines import NO_VARIANCE, inverse_distance, nearest_neighbour
from geocond.conventions import from_pykrige, to_gstools
from geocond.covariance import CovarianceComponent, CovarianceModel, principal_frame
from geocond.kriging import Observations, predict, support_covariance
from geocond.neighborhood import Neighborhood
from geocond.support import Support, block_support, line_support, point_support
from geocond.validation import ValidationError

REFERENCE = json.loads((Path(__file__).parent / "reference" / "gstat_fixed_covariance.json").read_text("utf-8"))
F = REFERENCE["fixture"]
TOL = 2e-10  # the tolerance the gstat reference itself declared


def lmc(nugget=True, cross=True, scale_secondary=1.0) -> CovarianceModel:
    """The reference LMC (gstat Exp a = 4 is GeoCond range 12; Sph 9 is 9), optionally without its nugget, without
    its cross terms, or with the secondary variable's units multiplied."""
    s = np.diag([1.0, scale_secondary])

    def mat(name):
        m = np.array(F["components"][name], dtype=float)
        if not cross:
            m[0, 1] = m[1, 0] = 0.0
        return s @ m @ s

    return CovarianceModel(
        (CovarianceComponent("exponential", 3.0 * F["ranges"][0], mat("exponential")),
         CovarianceComponent("spherical", F["ranges"][1], mat("spherical"))),
        nugget=mat("nugget") if nugget else np.zeros((2, 2)),
    )


def points(rows):
    return [point_support([r["x"], r["y"], r["z"]]) for r in rows]


def primary(error=None):
    return Observations(points(F["primary"]), [r["value"] for r in F["primary"]], error_variance=error)


def coupled(secondary_values=None, scale=1.0):
    values = [r["value"] for r in F["primary"]] + [scale * v for v in (
        secondary_values if secondary_values is not None else [r["value"] for r in F["secondary"]])]
    return Observations(points(F["primary"]) + points(F["secondary"]), values,
                        variables=[0] * len(F["primary"]) + [1] * len(F["secondary"]))


TARGETS = points(F["targets"])


@pytest.mark.parametrize("method,label", [("simple", "SK"), ("ordinary", "OK")])
def test_univariate_kriging_matches_gstat(method, label):
    ref = REFERENCE["univariate"][label]["gstat"]
    out = predict(primary(), TARGETS, lmc().variable(0), method=method,
                  mean=F["fixedMeans"][0] if method == "simple" else None)
    assert out.valid.all()
    assert np.allclose(out.mean, [r["var1.pred"] for r in ref], rtol=0, atol=TOL)
    assert np.allclose(out.variance, [r["var1.var"] for r in ref], rtol=0, atol=TOL)
    weights = [np.array(d["weights"])[:, 0] for d in out.diagnostics]
    recorded = [np.array(d["weights"]) for d in REFERENCE["univariate"][label]["reference"]["details"]]
    assert all(np.allclose(w, r, atol=1e-12) for w, r in zip(weights, recorded, strict=True))


@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_coupled_cokriging_matches_gstat_with_its_error_cross_covariance(method):
    ref = REFERENCE["coupled"][method]["gstat"]
    out = predict(coupled(), TARGETS, lmc(), method=method, target_variable=(0, 1),
                  mean=F["fixedMeans"] if method == "simple" else None)
    assert np.allclose(out.means[:, 0], [r["A.pred"] for r in ref], rtol=0, atol=TOL)
    assert np.allclose(out.means[:, 1], [r["B.pred"] for r in ref], rtol=0, atol=TOL)
    assert np.allclose(out.variances[:, 0], [r["A.var"] for r in ref], rtol=0, atol=TOL)
    assert np.allclose(out.variances[:, 1], [r["B.var"] for r in ref], rtol=0, atol=TOL)
    assert np.allclose(out.error_covariance[:, 0, 1], [r["cov.A.B"] for r in ref], rtol=0, atol=TOL)
    if method == "ordinary":
        for d in out.diagnostics:
            sums = d["weight_sums"]
            assert np.allclose(sums[0], [1.0, 0.0], atol=1e-12) and np.allclose(sums[1], [0.0, 1.0], atol=1e-12)
            assert d["constraint_residual"] < 1e-12 and d["linear_residual"] < 1e-12


@pytest.mark.parametrize("method,label", [("simple", "SK"), ("ordinary", "OK")])
def test_zero_cross_covariance_reduces_to_the_univariate_solution(method, label):
    mean = F["fixedMeans"] if method == "simple" else None
    out = predict(coupled(), TARGETS, lmc(cross=False), method=method, mean=mean)
    single = REFERENCE["univariate"][label]["gstat"]
    assert np.allclose(out.mean, [r["var1.pred"] for r in single], atol=TOL)
    assert np.allclose(out.variance, [r["var1.var"] for r in single], atol=TOL)
    assert np.allclose(out.mean, REFERENCE["coupled"][method]["zeroCrossPrediction"], atol=TOL)


@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_secondary_information_flows_into_the_primary_prediction(method):
    record = REFERENCE["coupled"][method]["secondaryPerturbation"]
    mean = F["fixedMeans"] if method == "simple" else None
    base = predict(coupled(), TARGETS, lmc(), method=method, mean=mean).mean
    changed = [r["value"] for r in F["secondary"]]
    changed[record["row"] - 1] += record["delta"]
    moved = predict(coupled(changed), TARGETS, lmc(), method=method, mean=mean).mean
    assert np.allclose(moved - base, record["primaryPredictionChange"], atol=TOL)


@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_secondary_units_do_not_change_the_primary_result(method):
    mean = F["fixedMeans"] if method == "simple" else None
    base = predict(coupled(), TARGETS, lmc(), method=method, target_variable=(0, 1), mean=mean)
    scaled_mean = [F["fixedMeans"][0], 1000 * F["fixedMeans"][1]] if method == "simple" else None
    scaled = predict(coupled(scale=1000.0), TARGETS, lmc(scale_secondary=1000.0), method=method,
                     target_variable=(0, 1), mean=scaled_mean)
    assert np.allclose(scaled.means[:, 0], base.means[:, 0], atol=1e-9)
    assert np.allclose(scaled.variances[:, 0], base.variances[:, 0], atol=1e-9)
    assert np.allclose(scaled.variances[:, 1] / 1e6, base.variances[:, 1], rtol=1e-9)


def offsets_support(center, measure):
    offsets = np.array(REFERENCE["support"]["offsets"], dtype=float)
    return Support(np.asarray(center) + offsets, np.full(len(offsets), 1 / len(offsets)), "weighted",
                   measure=measure)


def test_block_support_prediction_matches_gstat():
    ref = REFERENCE["support"]["gstat"]
    targets = [offsets_support([r["x"], r["y"], r["z"]], "continuous") for r in F["targets"]]
    out = predict(primary(), targets, lmc(nugget=False).variable(0), method="ordinary")
    assert np.allclose(out.mean, [r["var1.pred"] for r in ref], atol=TOL)
    assert np.allclose(out.variance, [r["var1.var"] for r in ref], atol=TOL)


def test_a_continuous_block_drops_the_nugget_and_a_finite_point_average_keeps_nugget_over_n():
    block = REFERENCE["support"]["continuousBlockNugget"]
    model = lmc().variable(0)
    cont = predict(primary(), [offsets_support([r["x"], r["y"], r["z"]], "continuous") for r in F["targets"]], model)
    disc = predict(primary(), [offsets_support([r["x"], r["y"], r["z"]], "discrete") for r in F["targets"]], model)
    assert np.allclose(cont.mean, [r["var1.pred"] for r in block["gstat"]], atol=TOL)
    assert np.allclose(cont.variance, [r["var1.var"] for r in block["gstat"]], atol=TOL)
    assert np.allclose(disc.variance, block["finitePointAverageVariance"], atol=TOL)
    assert np.allclose(disc.variance - cont.variance, block["finitePointAverageExtraNugget"], atol=1e-12)


def test_measurement_error_smooths_at_the_data_and_targets_the_latent_process():
    ref = REFERENCE["measurementError"]
    obs = primary(error=np.full(len(F["primary"]), ref["variance"]))
    out = predict(obs, points(F["primary"]), lmc(nugget=False).variable(0), method="simple", mean=F["fixedMeans"][0])
    assert np.allclose(out.mean, [r["var1.pred"] for r in ref["gstat"]], atol=TOL)
    assert np.allclose(out.variance, [r["var1.var"] for r in ref["gstat"]], atol=TOL)
    assert np.max(np.abs(out.mean - [r["value"] for r in F["primary"]])) > 0.01


def test_a_process_nugget_reproduces_the_data_exactly_at_their_supports():
    out = predict(primary(), points(F["primary"]), lmc().variable(0), method="simple", mean=F["fixedMeans"][0])
    assert np.allclose(out.mean, [r["value"] for r in F["primary"]], atol=1e-12)
    assert np.allclose(out.variance, 0.0, atol=1e-12)


# ------------------------------------------------------------------------------------------- acceptance cases


def spherical(sill=1.0, range_=30.0, nugget=0.0, frame=None, ranges=None):
    return CovarianceModel((CovarianceComponent("spherical", ranges if ranges is not None else range_, [[sill]], frame),),
                           nugget=[[nugget]])


@pytest.fixture
def scattered():
    rng = np.random.default_rng(12)
    x = rng.uniform(0, 60, size=(40, 3))
    return x, rng.uniform(0, 60, size=(12, 3))


def test_ordinary_kriging_reproduces_a_constant_field(scattered):
    x, t = scattered
    out = predict(Observations([point_support(p) for p in x], np.full(len(x), 3.25)), [point_support(p) for p in t],
                  spherical(nugget=0.1))
    assert np.allclose(out.mean, 3.25, atol=1e-12)


def test_universal_kriging_reproduces_a_linear_field_at_withheld_points(scattered):
    x, t = scattered
    coef = np.array([0.4, -1.2, 0.7])
    obs = Observations([point_support(p) for p in x], 5.0 + x @ coef)
    out = predict(obs, [point_support(p) for p in t], spherical(), method="universal", drift="linear")
    assert np.allclose(out.mean, 5.0 + t @ coef, atol=1e-9)
    assert all(d["constraint_residual"] < 1e-10 for d in out.diagnostics)


def test_a_zero_nugget_model_honours_the_data(scattered):
    x, _ = scattered
    z = np.sin(x[:, 0] / 9) + x[:, 2] / 30
    out = predict(Observations([point_support(p) for p in x], z), [point_support(p) for p in x[:5]], spherical())
    assert np.allclose(out.mean, z[:5], atol=1e-9) and np.allclose(out.variance, 0.0, atol=1e-9)


def test_rotation_and_translation_with_the_anisotropy_change_nothing(scattered):
    x, t = scattered
    z = np.cos(x[:, 1] / 11)
    frame = principal_frame(40, -20, 10)
    model = spherical(ranges=[50.0, 25.0, 12.0], frame=frame, nugget=0.05)
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    shift = np.array([1000.0, -250.0, 40.0])
    turned = spherical(ranges=[50.0, 25.0, 12.0], frame=q @ frame, nugget=0.05)
    a = predict(Observations([point_support(p) for p in x], z), [point_support(p) for p in t], model)
    b = predict(Observations([point_support(q @ p + shift) for p in x], z), [point_support(q @ p + shift) for p in t],
                turned)
    assert np.allclose(a.mean, b.mean, atol=1e-10) and np.allclose(a.variance, b.variance, atol=1e-10)


def test_scaling_the_values_and_the_model_scales_mean_and_variance(scattered):
    x, t = scattered
    z = x[:, 0] / 10
    a = predict(Observations([point_support(p) for p in x], z), [point_support(p) for p in t], spherical(nugget=0.2))
    b = predict(Observations([point_support(p) for p in x], 7 * z), [point_support(p) for p in t],
                spherical(sill=49.0, nugget=49 * 0.2))
    assert np.allclose(b.mean, 7 * a.mean, rtol=1e-10) and np.allclose(b.variance, 49 * a.variance, rtol=1e-10)


def test_uninformed_singular_and_duplicate_cases_report_an_honest_status():
    x = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    obs = Observations([point_support(p) for p in x], [1.0, 2.0, 3.0, 4.0])
    far = predict(obs, [point_support([500.0, 0, 0])], spherical(), neighborhood=Neighborhood(radius=50.0))
    assert far.status == ("uninformed",) and np.isnan(far.mean[0]) and "min_samples" in far.reasons[0]
    line = predict(obs, [point_support([15.0, 5, 0])], spherical(), method="universal", drift="linear")
    assert line.status == ("failed",) and "rank deficient" in line.reasons[0]
    with pytest.raises(ValidationError, match="same variable at the same support"):
        Observations([point_support([0, 0, 0]), point_support([0, 0, 0])], [1.0, 2.0])
    near = Observations([point_support([0, 0, 0]), point_support([1e-13, 0, 0]), point_support([10, 0, 0])], [1.0, 2.0, 3.0])
    singular = predict(near, [point_support([5.0, 1, 0])], spherical())
    assert singular.status == ("failed",) and "ill conditioned" in singular.reasons[0]
    rescued = predict(near, [point_support([5.0, 1, 0])], spherical(), solver={"jitter": 1e-6})
    assert rescued.valid[0] and rescued.diagnostics[0]["regularization"] > 0


def test_line_self_covariance_converges_at_second_order_to_the_exact_integral():
    """The covariance has a kink at zero separation, so a product Gauss rule converges algebraically on a support's
    self-covariance; the exact value for a line of length L is (2/L^2) int_0^L (L - h) C(h) dh."""
    length, a = 10.0, 40.0
    exact = 1 - 0.5 * length / a + 0.05 * (length / a) ** 3  # the spherical integral in closed form
    model = spherical(range_=a)
    errors = []
    for order in (8, 16, 32, 64, 128):
        s = line_support([0, 0, 0], [0, 0, -length], order=order)
        errors.append(abs(support_covariance(model, s, s) - exact))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.3 < r < 4.7 for r in ratios)  # second order: the error falls about fourfold per doubling
    assert errors[-1] < 1e-5


def test_block_quadrature_converges(scattered):
    x, _ = scattered
    obs = Observations([point_support(p) for p in x], np.sin(x[:, 0] / 7))
    results = [predict(obs, [block_support([30.0, 30, 30], [12.0, 8, 6], order=o)], spherical(range_=25.0))
               for o in (4, 8, 12, 16)]
    mean_steps = np.abs(np.diff([r.mean[0] for r in results]))
    var_steps = np.abs(np.diff([r.variance[0] for r in results]))
    assert np.all(np.diff(mean_steps) < 0) and np.all(np.diff(var_steps) < 0)
    assert mean_steps[-1] < 1e-6 and var_steps[-1] < 2e-5


# ---------------------------------------------------------------------------------------------- references


@pytest.mark.reference
@pytest.mark.parametrize("family", ["spherical", "exponential", "gaussian"])
def test_pykrige_ordinary_and_universal_3d_agree(scattered, family):
    ok3d = pytest.importorskip("pykrige.ok3d")
    uk3d = pytest.importorskip("pykrige.uk3d")
    x, t = scattered
    z = np.sin(x[:, 0] / 9) + 0.02 * x[:, 1]
    params = {"psill": 1.4, "range": 35.0, "nugget": 0.15}
    ours_model = from_pykrige(family, params["psill"], params["range"], params["nugget"])
    obs = Observations([point_support(p) for p in x], z)
    targets = [point_support(p) for p in t]
    ok = ok3d.OrdinaryKriging3D(*x.T, z, variogram_model=family, variogram_parameters=params)
    k, ss = ok.execute("points", *t.T)
    ours = predict(obs, targets, ours_model)
    assert np.allclose(ours.mean, k, atol=1e-8) and np.allclose(ours.variance, ss, atol=1e-8)
    uk = uk3d.UniversalKriging3D(*x.T, z, variogram_model=family, variogram_parameters=params,
                                 drift_terms=["regional_linear"])
    k, ss = uk.execute("points", *t.T)
    ours = predict(obs, targets, ours_model, method="universal", drift="linear")
    assert np.allclose(ours.mean, k, atol=1e-8) and np.allclose(ours.variance, ss, atol=1e-8)


@pytest.mark.reference
def test_gstools_simple_and_ordinary_kriging_agree(scattered):
    gs = pytest.importorskip("gstools")
    x, t = scattered
    z = np.cos(x[:, 2] / 8)
    model = from_pykrige("exponential", 2.0, 30.0, 0.0)
    obs = Observations([point_support(p) for p in x], z)
    targets = [point_support(p) for p in t]
    simple = gs.krige.Simple(to_gstools(model), cond_pos=list(x.T), cond_val=z, mean=0.3)
    field, var = simple(list(t.T), return_var=True)
    ours = predict(obs, targets, model, method="simple", mean=0.3)
    assert np.allclose(ours.mean, field, atol=1e-9) and np.allclose(ours.variance, var, atol=1e-9)
    ordinary = gs.krige.Ordinary(to_gstools(model), cond_pos=list(x.T), cond_val=z)
    field, var = ordinary(list(t.T), return_var=True)
    ours = predict(obs, targets, model)
    assert np.allclose(ours.mean, field, atol=1e-9) and np.allclose(ours.variance, var, atol=1e-9)


# ------------------------------------------------------------------------------------ neighbourhoods, baselines


def test_neighbourhood_limits_are_deterministic_and_group_aware():
    x = np.array([[i, 0.0, 0.0] for i in range(1, 9)])
    holes = np.array(["A", "A", "A", "B", "B", "C", "C", "C"])
    obs = Observations([point_support(p) for p in x], np.arange(8.0), groups=holes)
    hood = Neighborhood(max_samples=4, max_per_group=2)
    out = predict(obs, [point_support([0.0, 0, 0])], spherical(range_=50.0), neighborhood=hood)
    assert out.diagnostics[0]["ids"] == ["0", "1", "3", "4"]  # two from A, then two from B
    strict = predict(obs, [point_support([0.0, 0, 0])], spherical(range_=50.0),
                     neighborhood=Neighborhood(max_samples=2, min_groups=2))
    assert strict.status == ("uninformed",) and "distinct groups" in strict.reasons[0]
    aniso = Neighborhood(max_samples=1, metric=(np.eye(3), [100.0, 1.0, 1.0]))
    far_along = Observations([point_support([50.0, 0, 0]), point_support([0.0, 2.0, 0])], [1.0, 2.0])
    assert predict(far_along, [point_support([0.0, 0, 0])], spherical(range_=80.0), neighborhood=aniso).mean[0] == 1.0


def test_baselines_carry_no_variance():
    x = np.array([[0.0, 0, 0], [10.0, 0, 0], [0.0, 20.0, 0]])
    obs = Observations([point_support(p) for p in x], [1.0, 2.0, 4.0])
    nn = nearest_neighbour(obs, [point_support([6.0, 0, 0])])
    assert nn.mean[0] == 2.0 and np.isnan(nn.variance[0]) and nn.reasons[0] == NO_VARIANCE
    idw = inverse_distance(obs, [point_support([5.0, 0, 0])], power=2)
    d = np.array([5.0, 5.0, np.hypot(5, 20)])
    w = d**-2 / (d**-2).sum()
    assert idw.mean[0] == pytest.approx(w @ [1.0, 2.0, 4.0])
    assert inverse_distance(obs, [point_support([10.0, 0, 0])]).mean[0] == 2.0
