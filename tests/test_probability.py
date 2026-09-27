"""Isotonic correction against scikit-learn and a generic QP, and multiple-indicator kriging."""

import numpy as np
import pytest
from scipy.optimize import minimize

from geocond.covariance import CovarianceComponent, CovarianceModel
from geocond.kriging import Observations
from geocond.neighborhood import Neighborhood
from geocond.probability import bounded_isotonic, indicator_kriging, pava
from geocond.support import point_support


def test_pava_is_the_weighted_least_squares_nondecreasing_fit():
    rng = np.random.default_rng(3)
    for _ in range(50):
        y = rng.normal(size=12)
        w = rng.uniform(0.2, 3.0, size=12)
        fit = pava(y, w)
        assert np.all(np.diff(fit) >= -1e-15)
        # no nondecreasing sequence does better: compare with a generic constrained solve
        cons = [{"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]} for i in range(11)]
        qp = minimize(lambda x, w=w, y=y: np.sum(w * (x - y) ** 2), np.sort(y), constraints=cons, method="SLSQP",
                      options={"ftol": 1e-14, "maxiter": 500})
        assert np.sum(w * (fit - y) ** 2) <= qp.fun + 1e-9


def test_the_bounded_projection_is_pava_clipped_to_the_unit_interval():
    rng = np.random.default_rng(8)
    for _ in range(50):
        y = rng.uniform(-0.4, 1.4, size=8)
        fit = bounded_isotonic(y)
        cons = [{"type": "ineq", "fun": lambda x, i=i: x[i + 1] - x[i]} for i in range(7)]
        qp = minimize(lambda x, y=y: np.sum((x - y) ** 2), np.clip(np.sort(y), 0, 1), constraints=cons,
                      bounds=[(0, 1)] * 8, method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
        assert np.allclose(fit, qp.x, atol=1e-6)


@pytest.mark.reference
def test_pava_agrees_with_scikit_learn():
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(5)
    for _ in range(30):
        y = rng.normal(size=15)
        w = rng.uniform(0.5, 2.0, size=15)
        ref = isotonic.IsotonicRegression(increasing=True).fit(np.arange(15), y, sample_weight=w).predict(np.arange(15))
        assert np.allclose(pava(y, w), ref, atol=1e-12)
        bounded = isotonic.IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0).fit(np.arange(15), y).predict(np.arange(15))
        assert np.allclose(bounded_isotonic(y), bounded, atol=1e-12)


def indicator_models(ranges):
    return [CovarianceModel((CovarianceComponent("spherical", r, [[0.25]]),)) for r in ranges]


def test_indicator_kriging_reproduces_indicators_at_the_data_and_orders_the_cdf():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 50, size=(60, 3))
    z = np.sin(x[:, 0] / 8) + 0.3 * rng.normal(size=60)
    obs = Observations([point_support(p) for p in x], z)
    thresholds = np.quantile(z, [0.2, 0.4, 0.6, 0.8])
    result = indicator_kriging(obs, [point_support(p) for p in x[:6]], thresholds, indicator_models([15, 18, 18, 15]))
    assert result.valid.all()
    expected = (z[:6, None] <= thresholds[None, :]).astype(float)
    assert np.allclose(result.raw, expected, atol=1e-9) and np.allclose(result.cdf, expected, atol=1e-9)
    grid = [point_support(p) for p in rng.uniform(0, 50, size=(40, 3))]
    out = indicator_kriging(obs, grid, thresholds, indicator_models([6, 25, 9, 30]))
    assert np.all(np.diff(out.cdf, axis=1) >= -1e-12) and np.all((out.cdf >= 0) & (out.cdf <= 1))
    assert np.allclose(out.correction_max, np.max(np.abs(out.cdf - out.raw), axis=1))
    assert (out.correction_max > 1e-6).any()  # different ranges per threshold do produce order violations here


def test_probabilities_exist_between_thresholds_only():
    x = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]])
    obs = Observations([point_support(p) for p in x], [1.0, 2.0, 3.0, 4.0])
    result = indicator_kriging(obs, [point_support([15.0, 0, 0])], [1.5, 2.5, 3.5], indicator_models([20, 20, 20]))
    mid, status = result.probability_below(2.0)
    assert status == "interpolated" and result.cdf[0, 0] <= mid[0] <= result.cdf[0, 1]
    tail, status = result.probability_below(0.5)
    assert status == "unmodeled-tail" and np.isnan(tail[0])
    above, _ = result.probability_above(2.5)
    assert above[0] == pytest.approx(1 - result.cdf[0, 1])


def test_an_uninformed_neighbourhood_is_carried_through():
    x = np.array([[0.0, 0, 0], [10, 0, 0]])
    obs = Observations([point_support(p) for p in x], [1.0, 2.0])
    result = indicator_kriging(obs, [point_support([900.0, 0, 0])], [1.5], indicator_models([20]),
                               neighborhood=Neighborhood(radius=100.0))
    assert result.status == ("uninformed",) and "threshold 1.5" in result.reasons[0] and np.isnan(result.cdf[0, 0])
