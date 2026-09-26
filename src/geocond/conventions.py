"""Explicit range conventions of the independent reference libraries.

Libraries that share a model name do not share its range. GeoCond's families use the practical range: the
exponential and Gaussian correlations reach exp(-3) there and the spherical reaches zero. The conversions below were
measured by evaluating each library's own correlation on a separation grid (tests/test_conventions.py), never assumed
from the names:

| Library, model | Library correlation | GeoCond practical range R |
|---|---|---|
| gstat `Exp(a)` | exp(-h/a) | 3 a |
| gstat `Sph(a)` | 1 - 1.5 h/a + 0.5 (h/a)^3 | a |
| gstat `Gau(a)` | exp(-(h/a)^2) | sqrt(3) a |
| PyKrige exponential, range r | exp(-h/(r/3)) | r |
| PyKrige spherical, range r | 1 - 1.5 h/r + 0.5 (h/r)^3 | r |
| PyKrige gaussian, range r | exp(-(h/(4r/7))^2) | sqrt(3) 4r/7 |
| GSTools Exponential, len_scale l | exp(-h/l) | 3 l |
| GSTools Spherical, len_scale l | 1 - 1.5 h/l + 0.5 (h/l)^3 | l |
| GSTools Gaussian, len_scale l | exp(-(pi/4)(h/l)^2) | l sqrt(12/pi) |

The adapters cover isotropic univariate models, the form every library shares; anisotropic and multivariate models
are compared through evaluated covariance, never through parameter vectors.
"""

from math import pi, sqrt

from .covariance import CovarianceComponent, CovarianceModel
from .validation import ValidationError, positive

_GSTAT = {"Exp": ("exponential", 3.0), "Sph": ("spherical", 1.0), "Gau": ("gaussian", sqrt(3.0))}
_PYKRIGE = {"exponential": 1.0, "spherical": 1.0, "gaussian": sqrt(3.0) * 4.0 / 7.0}
_GSTOOLS = {"Exponential": ("exponential", 3.0), "Spherical": ("spherical", 1.0), "Gaussian": ("gaussian", sqrt(12.0 / pi))}


def from_gstat(rows: list[tuple[str, float, float]]) -> CovarianceModel:
    """A univariate isotropic model from gstat ``vgm`` rows ``(model, psill, range)``; ``Nug`` rows are the nugget."""
    nugget = 0.0
    components = []
    for model, psill, range_ in rows:
        psill = positive(psill, f"{model} psill", zero=True)
        if model == "Nug":
            nugget += psill
            continue
        if model not in _GSTAT:
            raise ValidationError(f"gstat model {model!r} has no GeoCond equivalent")
        family, factor = _GSTAT[model]
        components.append(CovarianceComponent(family, factor * positive(range_, f"{model} range"), [[psill]]))
    return CovarianceModel(tuple(components), nugget=[[nugget]])


def to_gstat(model: CovarianceModel) -> list[tuple[str, float, float]]:
    """gstat ``vgm`` rows for a univariate isotropic model."""
    _univariate_isotropic(model)
    rows = [("Nug", float(model.nugget[0, 0]), 0.0)] if model.nugget[0, 0] > 0 else []
    names = {family: (name, factor) for name, (family, factor) in _GSTAT.items()}
    for c in model.components:
        name, factor = names[c.family]
        rows.append((name, float(c.sill[0, 0]), float(c.ranges[0]) / factor))
    return rows


def from_pykrige(family: str, psill: float, range_: float, nugget: float = 0.0) -> CovarianceModel:
    """A model from PyKrige's ``[psill, range, nugget]`` for one of its three valid 3D covariance families (its
    one-dimensional hole-effect model is not a valid arbitrary 3D covariance and is refused)."""
    if family not in _PYKRIGE:
        raise ValidationError(f"PyKrige model {family!r} has no valid GeoCond 3D equivalent")
    component = CovarianceComponent(family, _PYKRIGE[family] * positive(range_, "range"), [[positive(psill, "psill", zero=True)]])
    return CovarianceModel((component,), nugget=[[positive(nugget, "nugget", zero=True)]])


def to_pykrige(model: CovarianceModel) -> tuple[str, list[float]]:
    """``(variogram_model, [psill, range, nugget])`` for a single-structure univariate isotropic model."""
    _univariate_isotropic(model)
    if len(model.components) != 1:
        raise ValidationError("PyKrige models have exactly one structure")
    c = model.components[0]
    return c.family, [float(c.sill[0, 0]), float(c.ranges[0]) / _PYKRIGE[c.family], float(model.nugget[0, 0])]


def to_gstools(model: CovarianceModel, dim: int = 3):
    """The GSTools model of a single-structure univariate isotropic model (GSTools is an optional dependency)."""
    import gstools

    _univariate_isotropic(model)
    if len(model.components) != 1:
        raise ValidationError("A GSTools covariance model has exactly one structure")
    c = model.components[0]
    name, factor = next((n, f) for n, (family, f) in _GSTOOLS.items() if family == c.family)
    return getattr(gstools, name)(
        dim=dim, var=float(c.sill[0, 0]), len_scale=float(c.ranges[0]) / factor, nugget=float(model.nugget[0, 0])
    )


def _univariate_isotropic(model: CovarianceModel) -> None:
    if model.n_variables != 1:
        raise ValidationError("the library adapters convert univariate models only")
    if not all(c.isotropic for c in model.components):
        raise ValidationError("the library adapters convert isotropic models only")
