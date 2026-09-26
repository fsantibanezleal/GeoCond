# Changelog

## [0.02.000] - 2026-09-26

### Added

- `covariance`: exponential, spherical and Gaussian families on the practical-range convention; geometric anisotropy
  through `principal_frame` (azimuth, dip, rake); `CovarianceModel` as a linear model of coregionalization whose sill
  matrices and nugget pass a full-spectrum PSD check; the process nugget at exactly zero separation; the direct model
  of one variable. Evaluated LMC covariances match R/gstat 2.1-6 within 2e-10 at seven lags.
- `conventions`: the measured range conventions of gstat, PyKrige and GSTools, with adapters for isotropic univariate
  models, checked against each library's own evaluated functions.
- `variogram`: direct and cross experimental variograms with stated membership rules (half-open lag bins, inclusive
  angle and bandwidth tolerances), downhole pairs, the classical and Cressie-Hawkins estimators, seeded pair sampling
  with its population recorded, retained pairs and cancellation; `fit_variogram`, bounded multi-start weighted least
  squares, isotropic or three principal ranges, with every start recorded. Bins are identical to an all-pair
  enumeration and to GSTools 1.7.0.
- `tests/reference/gstat_fixed_covariance.json`: the attributed gstat parity fixture (authored inputs, gstat and
  independent outputs) for this and the coming kriging and cokriging releases.
- Docs: the methods pages for covariance models and variograms, with figures and verified references; the README
  lists each capability with the reference it is checked against.

## [0.01.001] - 2026-09-26

### Added

- `publish-pypi.yml`: release-triggered trusted publishing to PyPI (pending publisher registered 2026-09-26); this is the first release published there.

## [Unreleased]

### Added
- Initial support-aware conditioning API and implementation contract for the first complete release.

