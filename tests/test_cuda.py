"""The PyTorch float64 lanes against the NumPy reference (run where PyTorch and a CUDA device exist)."""

import json
from pathlib import Path

import numpy as np
import pytest

from geocond.covariance import CovarianceComponent, CovarianceModel, principal_frame
from geocond.kriging import Observations, predict
from geocond.neighborhood import Neighborhood
from geocond.support import block_support, point_support
from geocond.validation import ValidationError
from geocond.variogram import experimental_cross_variogram, experimental_variogram

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")]

REFERENCE = json.loads((Path(__file__).parent / "reference" / "gstat_fixed_covariance.json").read_text("utf-8"))
F = REFERENCE["fixture"]


def lmc():
    return CovarianceModel(
        (CovarianceComponent("exponential", 12.0, F["components"]["exponential"]),
         CovarianceComponent("spherical", 9.0, F["components"]["spherical"])),
        nugget=F["components"]["nugget"],
    )


def points(rows):
    return [point_support([r["x"], r["y"], r["z"]]) for r in rows]


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 100, (1500, 3))
    return x, rng.normal(size=1500), rng.normal(size=1500)


@pytest.mark.parametrize("kw", [{}, {"direction": [1, 0.3, 0], "angle_tolerance": 30, "bandwidth": 12},
                                {"estimator": "cressie-hawkins"}])
def test_the_torch_pair_lane_gives_the_reference_bins(cloud, kw):
    x, z, _ = cloud
    edges = np.linspace(0, 60, 13)
    a = experimental_variogram(x, z, edges, **kw)
    b = experimental_variogram(x, z, edges, backend="torch", **kw)
    assert np.array_equal(a.counts, b.counts) and a.coincident_pairs == b.coincident_pairs
    assert np.allclose(a.values, b.values, rtol=1e-12, atol=0, equal_nan=True)
    assert np.allclose(a.separation, b.separation, rtol=1e-12, atol=0, equal_nan=True)
    assert sorted(zip(a.pair_i.tolist(), a.pair_j.tolist(), strict=True)) == sorted(
        zip(b.pair_i.tolist(), b.pair_j.tolist(), strict=True))


def test_the_torch_cross_variogram_and_its_limits(cloud):
    x, z, w = cloud
    edges = np.linspace(0, 60, 7)
    a = experimental_cross_variogram(x, z, w, edges)
    b = experimental_cross_variogram(x, z, w, edges, backend="torch")
    assert np.array_equal(a.counts, b.counts) and np.allclose(a.values, b.values, rtol=1e-12, equal_nan=True)
    with pytest.raises(ValidationError, match="numpy"):
        experimental_variogram(x, z, edges, backend="torch", max_pairs=100)


@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_batched_cokriging_matches_the_reference_and_gstat(method):
    obs = Observations(points(F["primary"]) + points(F["secondary"]),
                       [r["value"] for r in F["primary"]] + [r["value"] for r in F["secondary"]],
                       variables=[0] * 6 + [1] * 7)
    mean = F["fixedMeans"] if method == "simple" else None
    a = predict(obs, points(F["targets"]), lmc(), method=method, target_variable=(0, 1), mean=mean)
    b = predict(obs, points(F["targets"]), lmc(), method=method, target_variable=(0, 1), mean=mean, backend="torch")
    assert np.allclose(a.means, b.means, atol=1e-12) and np.allclose(a.error_covariance, b.error_covariance, atol=1e-12)
    ref = REFERENCE["coupled"][method]["gstat"]
    assert np.allclose(b.means[:, 0], [r["A.pred"] for r in ref], atol=2e-10)
    if method == "ordinary":
        assert all(np.allclose(d["weight_sums"][0], [1.0, 0.0], atol=1e-12) for d in b.diagnostics)


def test_neighbourhoods_statuses_and_measurement_error_agree():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 300, (600, 3))
    frame = principal_frame(30, -10, 5)
    model = CovarianceModel((CovarianceComponent("spherical", [150.0, 80.0, 30.0], [[1.0]], frame),), nugget=[[0.1]])
    z = np.sin(x[:, 0] / 25)
    obs = Observations([point_support(p) for p in x], z, error_variance=np.full(600, 0.02))
    targets = [point_support(p) for p in np.r_[rng.uniform(0, 300, (200, 3)), [[5000.0, 0, 0]]]]
    hood = Neighborhood(max_samples=24, radius=120.0)
    a = predict(obs, targets, model, neighborhood=hood)
    b = predict(obs, targets, model, neighborhood=hood, backend="torch")
    assert a.status == b.status and b.status[-1] == "uninformed"
    ok = a.valid
    assert np.allclose(a.mean[ok], b.mean[ok], atol=1e-11) and np.allclose(a.variance[ok], b.variance[ok], atol=1e-11)
    near = Observations([point_support([0, 0, 0]), point_support([1e-13, 0, 0]), point_support([10, 0, 0])],
                        [1.0, 2.0, 3.0])
    sharp = CovarianceModel((CovarianceComponent("spherical", 300.0, [[1.0]]),))  # no nugget: near-duplicates are singular
    for backend in ("numpy", "torch"):
        assert predict(near, [point_support([5.0, 1, 0])], sharp, backend=backend).status == ("failed",)
    assert predict(near, [point_support([5.0, 1, 0])], model, backend="torch").status == ("estimated",)  # a nugget conditions them


def test_the_torch_lane_refuses_what_it_does_not_implement():
    obs = Observations([point_support([0, 0, 0]), point_support([10, 0, 0])], [1.0, 2.0])
    model = CovarianceModel((CovarianceComponent("spherical", 30.0, [[1.0]]),))
    with pytest.raises(ValidationError, match="point supports"):
        predict(obs, [block_support([5, 0, 0], [2, 2, 2])], model, backend="torch")
    with pytest.raises(ValidationError, match="simple and ordinary"):
        predict(obs, [point_support([5, 0, 0])], model, method="universal", drift="linear", backend="torch")
