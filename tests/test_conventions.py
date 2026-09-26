"""The range conventions of gstat, PyKrige and GSTools, measured on their evaluated functions."""

import numpy as np
import pytest

from geocond.conventions import from_gstat, from_pykrige, to_gstat, to_gstools, to_pykrige
from geocond.validation import ValidationError

LAGS = np.array([0.0, 0.3, 1.0, 2.5, 4.0, 7.5, 12.0, 30.0])
H = LAGS[:, None] * np.array([0.6, 0.8, 0.0])


@pytest.mark.parametrize("model,range_", [("Exp", 4.0), ("Sph", 9.0), ("Gau", 3.0)])
def test_gstat_rows_convert_to_the_curve_gstat_evaluates(model, range_):
    ours = from_gstat([("Nug", 0.2, 0.0), (model, 1.3, range_)])
    h = LAGS
    gstat_curve = {
        "Exp": 1.3 * np.exp(-h / range_),
        "Sph": 1.3 * np.where(h < range_, 1 - 1.5 * h / range_ + 0.5 * (h / range_) ** 3, 0.0),
        "Gau": 1.3 * np.exp(-((h / range_) ** 2)),
    }[model] + np.where(h == 0, 0.2, 0.0)
    assert np.allclose(ours.covariance(H), gstat_curve, atol=1e-14)
    assert to_gstat(ours) == [("Nug", 0.2, 0.0), (model, 1.3, pytest.approx(range_))]


@pytest.mark.reference
@pytest.mark.parametrize("family", ["exponential", "spherical", "gaussian"])
def test_pykrige_parameters_convert_to_pykriges_own_variogram(family):
    vm = pytest.importorskip("pykrige.variogram_models")
    params = [1.7, 6.0, 0.25]
    library = getattr(vm, f"{family}_variogram_model")(params, LAGS[1:])
    ours = from_pykrige(family, *params)
    assert np.allclose(ours.variogram(H[1:]), library, atol=1e-13)
    assert to_pykrige(ours) == (family, pytest.approx(params))


@pytest.mark.reference
@pytest.mark.parametrize("family", ["exponential", "spherical", "gaussian"])
def test_gstools_models_evaluate_the_same_covariance(family):
    pytest.importorskip("gstools")
    ours = from_pykrige(family, 2.0, 8.0, 0.0)
    library = to_gstools(ours)
    assert np.allclose(library.covariance(LAGS), ours.covariance(H), atol=1e-13)


def test_the_hole_effect_and_multivariate_models_are_refused():
    with pytest.raises(ValidationError):
        from_pykrige("hole-effect", 1.0, 1.0)
    with pytest.raises(ValidationError):
        from_gstat([("Mat", 1.0, 1.0)])
