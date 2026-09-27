# Changelog

## [0.03.000] - 2026-09-26

### Added

- `kriging.predict`: simple, ordinary and universal kriging and coupled simple and ordinary cokriging (traditional
  constraints) in one constrained solver, predicting one variable or several jointly with the full error covariance;
  covariance integrated over each observation's and target's support, the process nugget kept only between discrete
  supports at coincident nodes (gstat's continuous-block convention) and measurement error on the observation
  covariance only; drift averaged over supports and centred and scaled, with rank-deficient drifts failing instead of
  falling back to ordinary kriging; a condition limit of 1e12 and an explicit jitter policy; per-target statuses,
  reasons and diagnostics. `Observations` refuses undeclared exact duplicates.
- `neighborhood.Neighborhood`: deterministic distance ranking under an optional anisotropic metric, radius,
  per-variable and total limits, per-hole limits and minimum distinct holes, with uninformed statuses.
- `baselines`: nearest-neighbour and inverse-distance estimates with no implied variance.
- Tests: every case of the gstat fixture (univariate, coupled with cross covariance, secondary perturbation, zero-cross
  reduction, block support, continuous and finite-average nugget, measurement error, exact reproduction) within 2e-10;
  PyKrige 3D ordinary and universal kriging and GSTools simple and ordinary kriging; the dossier's acceptance cases;
  second-order quadrature convergence against the exact line integral.
- Docs: the kriging and cokriging methods page, with a support-integration figure.

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

