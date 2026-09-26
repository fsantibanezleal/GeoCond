"""Experimental variograms against an all-pair oracle and GSTools, and deterministic bounded fitting."""

import math

import numpy as np
import pytest

from geocond.covariance import CovarianceComponent, CovarianceModel, principal_frame
from geocond.validation import CancelledError, ValidationError
from geocond.variogram import (
    ExperimentalVariogram,
    experimental_cross_variogram,
    experimental_variogram,
    fit_variogram,
)


def oracle(x, z, edges, direction=None, tolerance=None, bandwidth=None, estimator="classical", w=None):
    """Every unordered pair enumerated in plain Python, with the documented membership rules."""
    nb = len(edges) - 1
    counts = [0] * nb
    sums = [0.0] * nb
    roots = [0.0] * nb
    seps = [0.0] * nb
    d = None if direction is None else np.asarray(direction, float) / np.linalg.norm(direction)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            h = x[j] - x[i]
            dist = math.sqrt(float(h @ h))
            if dist == 0:
                continue
            if d is not None:
                along = abs(float(h @ d))
                if tolerance is not None and along < dist * math.cos(math.radians(tolerance)) - 1e-12 * dist:
                    continue
                if bandwidth is not None and math.sqrt(max(dist**2 - along**2, 0)) > bandwidth + 1e-12 * max(dist, 1):
                    continue
            k = next((k for k in range(nb) if edges[k] <= dist < edges[k + 1]), None)
            if k is None:
                continue
            counts[k] += 1
            seps[k] += dist
            second = z if w is None else w
            sums[k] += (z[i] - z[j]) * (second[i] - second[j])
            roots[k] += math.sqrt(abs(z[i] - z[j]))
    values = []
    for k in range(nb):
        n = counts[k]
        if n == 0:
            values.append(math.nan)
        elif estimator == "classical":
            values.append(sums[k] / (2 * n))
        else:
            values.append((roots[k] / n) ** 4 / (2 * (0.457 + 0.494 / n + 0.045 / n**2)))
    with np.errstate(invalid="ignore"):
        return np.array(counts), np.array(seps) / np.array(counts), np.array(values)


@pytest.fixture
def cloud():
    rng = np.random.default_rng(7)
    return rng.uniform(0, 40, size=(45, 3)), rng.normal(size=45)


@pytest.mark.parametrize("estimator", ["classical", "cressie-hawkins"])
def test_omnidirectional_bins_match_the_all_pair_oracle(cloud, estimator):
    x, z = cloud
    edges = np.linspace(0, 35, 8)
    v = experimental_variogram(x, z, edges, estimator=estimator)
    counts, seps, values = oracle(x, z, edges, estimator=estimator)
    assert v.counts.tolist() == counts.tolist()
    assert np.allclose(v.separation, seps, equal_nan=True, rtol=1e-12)
    assert np.allclose(v.values, values, equal_nan=True, rtol=1e-12)
    assert len(v.pair_i) == counts.sum() and np.all(v.pair_i != v.pair_j)


def test_directional_bins_match_the_oracle(cloud):
    x, z = cloud
    edges = np.linspace(0, 35, 6)
    d = [1.0, 0.4, -0.3]
    v = experimental_variogram(x, z, edges, direction=d, angle_tolerance=25, bandwidth=6)
    counts, _, values = oracle(x, z, edges, d, 25, 6)
    assert v.counts.tolist() == counts.tolist()
    assert np.allclose(v.values, values, equal_nan=True, rtol=1e-12)


def test_pairs_exactly_on_the_boundaries_follow_the_stated_rules():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    v = experimental_variogram(x, [0.0, 1.0, 3.0], edges)
    # |h| = 2 is on an interior edge: upper bin; |h| = 3 is the last edge: no bin; |h| = 1: bin 1
    assert v.counts.tolist() == [0, 1, 1]
    square = np.array([[0.0, 0, 0], [1.0, 1.0, 0]])
    inside = experimental_variogram(square, [0.0, 1.0], [0.0, 5.0], direction=[1, 0, 0], angle_tolerance=45)
    outside = experimental_variogram(square, [0.0, 1.0], [0.0, 5.0], direction=[1, 0, 0], angle_tolerance=44.999)
    assert inside.counts.tolist() == [1] and outside.counts.tolist() == [0]
    band = np.array([[0.0, 0, 0], [10.0, 2.0, 0]])
    assert experimental_variogram(band, [0.0, 1.0], [0.0, 20.0], direction=[1, 0, 0], bandwidth=2.0).counts[0] == 1
    assert experimental_variogram(band, [0.0, 1.0], [0.0, 20.0], direction=[1, 0, 0], bandwidth=1.999).counts[0] == 0


def test_rotating_data_and_direction_together_leaves_the_estimate_unchanged(cloud):
    x, z = cloud
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    d = np.array([0.3, 1.0, 0.2])
    edges = np.linspace(0, 35, 6)
    a = experimental_variogram(x, z, edges, direction=d, angle_tolerance=30, bandwidth=10)
    b = experimental_variogram(x @ q.T, z, edges, direction=q @ d, angle_tolerance=30, bandwidth=10)
    assert a.counts.tolist() == b.counts.tolist()
    assert np.allclose(a.values, b.values, equal_nan=True, rtol=1e-12)


def test_planar_constant_duplicate_and_empty_cases_are_honest():
    rng = np.random.default_rng(4)
    planar = np.c_[rng.uniform(0, 10, size=(20, 2)), np.zeros(20)]
    assert experimental_variogram(planar, rng.normal(size=20), [0, 5, 10]).counts.sum() > 0
    constant = experimental_variogram(planar, np.full(20, 3.0), [0, 5, 15])
    assert np.all(constant.values[constant.valid] == 0)
    dup = np.array([[0.0, 0, 0], [0.0, 0, 0], [5.0, 0, 0]])
    v = experimental_variogram(dup, [1.0, 2.0, 3.0], [0, 10, 20])
    assert v.coincident_pairs == 1 and v.counts.tolist() == [2, 0]
    assert math.isnan(v.values[1]) and math.isnan(v.separation[1])


def test_downhole_pairs_stay_in_one_hole_and_use_measured_depth():
    x = np.array([[0, 0, -1.0], [0, 0, -2.0], [0, 0, -3.0], [50, 0, -1.0], [50, 0, -2.0]])
    holes = np.array(["A", "A", "A", "B", "B"])
    depths = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
    v = experimental_variogram(x, [1.0, 2.0, 4.0, 0.0, 1.0], [0.5, 1.5, 2.5], groups=holes, downhole=True, depths=depths)
    assert v.counts.tolist() == [3, 1]  # A: (1,2),(2,3), B: (1,2) at 1 m; A: (1,3) at 2 m
    assert np.all(holes[v.pair_i] == holes[v.pair_j])
    assert v.values[1] == pytest.approx((1.0 - 4.0) ** 2 / 2)
    with pytest.raises(ValidationError):
        experimental_variogram(x, np.zeros(5), [0, 1], groups=holes, downhole=True, depths=depths, direction=[1, 0, 0])


def test_the_cross_variogram_is_symmetric_and_reduces_to_the_direct_one(cloud):
    x, z = cloud
    w = 0.5 * z + np.random.default_rng(9).normal(size=len(z))
    edges = np.linspace(0, 35, 6)
    ab = experimental_cross_variogram(x, z, w, edges)
    ba = experimental_cross_variogram(x, w, z, edges)
    assert np.allclose(ab.values, ba.values, equal_nan=True)
    assert np.allclose(experimental_cross_variogram(x, z, z, edges).values, experimental_variogram(x, z, edges).values,
                       equal_nan=True)
    assert np.allclose(ab.values, oracle(x, z, edges, w=w)[2], equal_nan=True, rtol=1e-12)
    with pytest.raises(ValidationError, match="same rows"):
        experimental_cross_variogram(x, z, w[:-1], edges)


def test_pair_sampling_is_seeded_uniform_and_recorded(cloud):
    x, z = cloud
    edges = np.linspace(0, 70, 8)
    full = experimental_variogram(x, z, edges)
    a = experimental_variogram(x, z, edges, max_pairs=300, seed=1)
    b = experimental_variogram(x, z, edges, max_pairs=300, seed=1)
    c = experimental_variogram(x, z, edges, max_pairs=300, seed=2)
    assert a.sampled and a.population_pairs == 45 * 44 // 2 and a.seed == 1 and not full.sampled
    assert np.array_equal(a.pair_i, b.pair_i) and np.array_equal(a.pair_j, b.pair_j)
    assert not np.array_equal(a.pair_i, c.pair_i)
    assert a.counts.sum() == 300 - a.coincident_pairs
    population = set(zip(np.minimum(full.pair_i, full.pair_j).tolist(), np.maximum(full.pair_i, full.pair_j).tolist()))
    assert set(zip(np.minimum(a.pair_i, a.pair_j).tolist(), np.maximum(a.pair_i, a.pair_j).tolist())) <= population


def test_cancellation_stops_before_a_result():
    x = np.random.default_rng(0).uniform(size=(600, 3))
    with pytest.raises(CancelledError):
        experimental_variogram(x, np.zeros(600), [0, 1], cancel=lambda: True)


@pytest.mark.reference
@pytest.mark.parametrize("estimator,gs_name", [("classical", "matheron"), ("cressie-hawkins", "cressie")])
def test_gstools_gives_the_same_bins(estimator, gs_name):
    gs = pytest.importorskip("gstools")
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 50, size=(120, 3))
    z = rng.normal(size=120)
    edges = np.linspace(0, 30, 7)
    d = np.array([1.0, 0.5, 0.2]) / np.linalg.norm([1.0, 0.5, 0.2])
    for kwargs, gs_kwargs in [({}, {}), ({"direction": d, "angle_tolerance": 30, "bandwidth": 8},
                                         {"direction": [d], "angles_tol": np.radians(30), "bandwidth": 8})]:
        ours = experimental_variogram(x, z, edges, estimator=estimator, **kwargs)
        _, gamma, counts = gs.vario_estimate(x.T, z, bin_edges=edges, return_counts=True, estimator=gs_name,
                                             mesh_type="unstructured", **gs_kwargs)
        assert ours.counts.tolist() == np.asarray(counts).ravel().tolist()
        assert np.allclose(ours.values, np.asarray(gamma).ravel(), rtol=1e-12, equal_nan=True)


def synthetic(model, direction, edges, count=200):
    """A noise-free experimental variogram: the model's semivariance at each bin's midpoint."""
    edges = np.asarray(edges, float)
    sep = 0.5 * (edges[1:] + edges[:-1])
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    values = model.variogram(sep[:, None] * d[None, :])
    empty = np.zeros(0, np.int64)
    return ExperimentalVariogram(edges, sep, np.full(len(sep), count), values, empty, empty, empty, "classical", d,
                                 None, None, False, False, 0, False, None, 0)


def test_a_noise_free_nested_isotropic_variogram_is_recovered():
    truth = CovarianceModel(
        (CovarianceComponent("spherical", 30.0, [[1.0]]), CovarianceComponent("exponential", 90.0, [[0.5]])),
        nugget=[[0.2]],
    )
    v = synthetic(truth, [1, 0, 0], np.linspace(0, 150, 31))
    fit = fit_variogram(v, ["spherical", "exponential"], isotropic=True, starts=6)
    assert fit.objective < 1e-12
    m = fit.model
    assert m.nugget[0, 0] == pytest.approx(0.2, rel=1e-4)
    assert [c.sill[0, 0] for c in m.components] == pytest.approx([1.0, 0.5], rel=1e-4)
    assert [c.ranges[0] for c in m.components] == pytest.approx([30.0, 90.0], rel=1e-4)
    assert len(fit.starts) == 6 and fit.starts[fit.best_start]["objective"] == fit.objective


def test_three_principal_ranges_are_recovered_from_three_directions():
    frame = principal_frame(35, -15, 20)
    truth = CovarianceModel((CovarianceComponent("spherical", [60.0, 25.0, 10.0], [[1.3]], frame),), nugget=[[0.1]])
    variograms = [synthetic(truth, frame[:, axis], np.linspace(0, 80, 33)) for axis in range(3)]
    fit = fit_variogram(variograms, ["spherical"], rotation=frame, starts=5)
    assert fit.objective < 1e-12
    assert fit.model.components[0].ranges == pytest.approx([60.0, 25.0, 10.0], rel=1e-4)
    assert fit.model.components[0].sill[0, 0] == pytest.approx(1.3, rel=1e-4)
    with pytest.raises(ValidationError, match="spanning three dimensions"):
        fit_variogram(variograms[:2], ["spherical"], rotation=frame)


def test_fitting_is_deterministic():
    truth = CovarianceModel((CovarianceComponent("exponential", 40.0, [[2.0]]),), nugget=[[0.3]])
    v = synthetic(truth, [0, 1, 0], np.linspace(0, 100, 21))
    a = fit_variogram(v, ["exponential"], isotropic=True)
    b = fit_variogram(v, ["exponential"], isotropic=True)
    assert a.starts == b.starts and a.objective == b.objective


def test_a_field_simulated_from_a_known_covariance_fits_near_it():
    """A statistical sanity check (seeded, so deterministic): an exponential field with a nugget, 350 points."""
    truth = CovarianceModel((CovarianceComponent("exponential", 24.0, [[1.0]]),), nugget=[[0.15]])
    rng = np.random.default_rng(21)
    x = rng.uniform(0, 100, size=(350, 3)) * [1, 1, 0.3]
    c = truth.between(x, x) + 1e-10 * np.eye(len(x))
    z = np.linalg.cholesky(c) @ rng.normal(size=len(x))
    v = experimental_variogram(x, z, np.linspace(0, 45, 16))
    fit = fit_variogram(v, ["exponential"], isotropic=True)
    sill = fit.model.total_sill[0, 0]
    assert sill == pytest.approx(1.15, rel=0.3)
    assert fit.model.components[0].ranges[0] == pytest.approx(24.0, rel=0.45)
