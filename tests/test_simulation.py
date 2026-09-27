"""The normal-score table and sequential Gaussian simulation against the dense Gaussian conditional."""

import numpy as np
import pytest
from scipy.special import ndtri

from geocond.covariance import CovarianceComponent, CovarianceModel
from geocond.kriging import Observations
from geocond.neighborhood import Neighborhood
from geocond.simulation import NormalScoreTransform, sequential_gaussian
from geocond.support import point_support
from geocond.validation import ValidationError


def test_normal_scores_use_the_mid_rank_plotting_position():
    z = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    t = NormalScoreTransform.fit(z)
    assert np.allclose(t.values, [1, 2, 3, 4, 5])
    assert np.allclose(t.scores, ndtri((np.arange(1, 6) - 0.5) / 5))
    assert np.allclose(t.backward(t.forward(z)), z)


def test_ties_share_one_score_and_weights_move_the_positions():
    t = NormalScoreTransform.fit([1.0, 2.0, 2.0, 3.0])
    assert len(t.values) == 3 and t.probabilities.tolist() == pytest.approx([0.125, 0.5, 0.875])
    weighted = NormalScoreTransform.fit([1.0, 2.0, 3.0], weights=[2.0, 1.0, 1.0])
    assert weighted.probabilities.tolist() == pytest.approx([0.25, 0.625, 0.875])
    with pytest.raises(ValidationError, match="constant"):
        NormalScoreTransform.fit([2.0, 2.0, 2.0])


def test_tails_are_bounded_by_default_and_extend_only_when_declared():
    t = NormalScoreTransform.fit(np.arange(1.0, 11.0))
    assert t.tail == "bounded" and t.backward([-9.0])[0] == 1.0 and t.backward([9.0])[0] == 10.0
    wide = NormalScoreTransform.fit(np.arange(1.0, 11.0), lower_tail=(-4.0, 0.0), upper_tail=(4.0, 14.0))
    assert wide.tail == "declared" and wide.backward([-4.0])[0] == 0.0 and wide.backward([4.0])[0] == 14.0
    assert 0.0 < wide.backward([-2.5])[0] < 1.0
    with pytest.raises(ValidationError):
        NormalScoreTransform.fit(np.arange(1.0, 11.0), lower_tail=(0.0, 0.5))


def small_problem():
    model = CovarianceModel((CovarianceComponent("exponential", 30.0, [[1.0]]),))
    data = np.array([[0.0, 0, 0], [20, 5, 0], [8, 18, -3]])
    z = np.array([0.4, 1.9, 1.1])
    nodes = np.array([[5.0, 4, 0], [14, 9, -1], [3, 12, -2], [25, 15, 1], [11, 2, 2]])
    return model, data, z, nodes


def identity_transform(z):
    """A transform whose table is the identity over a wide range, so normal-score space equals native space."""
    grid = np.linspace(-8, 8, 2001)
    t = NormalScoreTransform.fit(grid)
    return NormalScoreTransform(grid, grid, t.probabilities, "bounded")


def test_full_neighbourhood_sgs_equals_the_dense_conditional_cholesky_draw():
    """With every datum and every earlier node in the neighbourhood, the sequential draw in path order is exactly
    mu + L eps, where L is the Cholesky factor of the dense conditional covariance ordered by the path."""
    model, data, z, nodes = small_problem()
    obs = Observations([point_support(p) for p in data], z)
    transform = identity_transform(z)
    result = sequential_gaussian(obs, [point_support(p) for p in nodes], model, transform, realizations=3, seed=7)
    Coo = model.between(data, data)
    Cto = model.between(nodes, data)
    Ctt = model.between(nodes, nodes)
    mu = Cto @ np.linalg.solve(Coo, z)
    sigma = Ctt - Cto @ np.linalg.solve(Coo, Cto.T)
    for r in range(3):
        path = result.paths[r]
        L = np.linalg.cholesky(sigma[np.ix_(path, path)])
        expected = mu[path] + L @ result.innovations[r, path]
        assert np.allclose(result.gaussian[r, path], expected, atol=1e-10)


def test_ensemble_moments_match_the_dense_conditional():
    model, data, z, nodes = small_problem()
    obs = Observations([point_support(p) for p in data], z)
    result = sequential_gaussian(obs, [point_support(p) for p in nodes], model, identity_transform(z),
                                 realizations=4000, seed=1)
    Coo = model.between(data, data)
    Cto = model.between(nodes, data)
    mu = Cto @ np.linalg.solve(Coo, z)
    sigma = model.between(nodes, nodes) - Cto @ np.linalg.solve(Coo, Cto.T)
    assert np.allclose(result.gaussian.mean(axis=0), mu, atol=0.05)
    assert np.allclose(np.cov(result.gaussian.T), sigma, atol=0.06)


def test_hard_data_are_honoured_and_seeds_reproduce():
    model, data, z, nodes = small_problem()
    obs = Observations([point_support(p) for p in data], z)
    t = NormalScoreTransform.fit(np.r_[z, 0.0, 3.0])
    targets = [point_support(p) for p in np.r_[nodes, data[:1]]]
    a = sequential_gaussian(obs, targets, model, t, realizations=4, seed=3)
    b = sequential_gaussian(obs, targets, model, t, realizations=4, seed=3)
    c = sequential_gaussian(obs, targets, model, t, realizations=4, seed=4)
    assert a.hard.tolist() == [False] * 5 + [True]
    assert np.all(a.native[:, -1] == z[0])
    assert np.array_equal(a.native, b.native) and np.array_equal(a.paths, b.paths)
    assert not np.array_equal(a.native, c.native)
    assert np.all((a.native >= 0.0) & (a.native <= 3.0))  # the bounded tail keeps the training extrema


def test_the_etype_is_the_native_ensemble_mean_not_a_backtransformed_gaussian_mean():
    model, data, z, nodes = small_problem()
    obs = Observations([point_support(p) for p in data], np.exp(z))  # skewed native values
    t = NormalScoreTransform.fit(np.exp(np.r_[z, -1.0, 2.5, 0.2, 0.9]))
    result = sequential_gaussian(obs, [point_support(p) for p in nodes], model, t, realizations=200, seed=2)
    assert np.allclose(result.etype(), result.native.mean(axis=0))
    assert not np.allclose(result.etype(), t.backward(result.gaussian.mean(axis=0)), atol=1e-3)
    assert result.exceedance(np.exp(1.0)).shape == (5,) and result.quantile(0.5).shape == (5,)


def test_a_limited_neighbourhood_still_conditions_on_data_and_earlier_nodes():
    model, data, z, nodes = small_problem()
    obs = Observations([point_support(p) for p in data], z)
    result = sequential_gaussian(obs, [point_support(p) for p in nodes], model, identity_transform(z),
                                 realizations=2, seed=5, neighborhood=Neighborhood(max_samples=3))
    assert np.isfinite(result.gaussian).all() and np.all(result.conditional_sd > 0)
