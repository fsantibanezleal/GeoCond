# Changelog

## [0.08.000] - 2026-09-27

### Added

- `direct_sampling(..., zones=(ti_zones, grid_zones))`: each node scans only the valid training-image centres of its
  own zone, in a seeded permutation of that zone, with the scan fraction applied to the zone's count (Mariethoz,
  Renard and Straubhaar 2010, section 6). One zone everywhere is the unzoned engine bit for bit; the PyTorch backend
  selects the same candidates with zones. On a layered prior 20 layers deep, zones keep the order (0.957 of cells
  against 0.959 in the TI) where the unzoned engine loses it (0.161). Needed by Sondara's categorical simulation in
  depth layers.

## [0.07.000] - 2026-09-27

### Added

- `sequential_gaussian(..., node_neighborhood=...)`: a two-part search, as GSLIB's `sstrat = 0`. The original data
  are selected by `neighborhood` with the observations' group identities (a per-group cap and the minimum apply to
  them), and the nodes already simulated by `node_neighborhood` (GSLIB's `ncnode`). Without it, the single search over
  data and nodes is unchanged, and group limits are now refused with a message instead of failing inside the search.
  Found by Sondara: along held-out drillholes, dense simulated nodes crowded the training holes out of a single
  search of 24; 24 data and 12 nodes lowered the E-type RMSE of Rocklea Fe from 16.42 to 15.11 wt%.
- `SimulationResult.node_neighborhood` records the node part of the search.

## [0.06.002] - 2026-09-27

### Fixed

- The linear-residual diagnostic of a kriging solve divided zero by zero, and reported NaN with a RuntimeWarning,
  when simple kriging's target lies beyond the range of every selected observation (zero weights, zero right-hand
  side). That exact solve now reports a residual of zero, in the NumPy and the torch lanes. Found by Sondara's
  spatial-margin predictions.

## [0.06.001] - 2026-09-26

### Added

- Minimum-curvature parity with welleng 0.29.1, the independent reference the API contract names: 12 seeded random
  surveys and a near-straight section wrapping through north, stations and arc interpolation within 1e-9 of the
  path length, doglegs within 1e-9 degrees. CI installs welleng.
- `docs/methods/06_geometry_and_compositing.md`: the convention, minimum curvature with arc interpolation and
  explicit extensions, supports and quadrature, and compositing, with two figures computed by GeoCond.

## [0.06.000] - 2026-09-26

### Added

- `cuda`: the float64 PyTorch lanes of the settled engine table. `experimental_variogram` and
  `experimental_cross_variogram` take `backend="torch"`: every spatial pair tiled on the device with the same
  membership rules; identical counts and pairs, bins within 1e-12 relative. `predict` takes `backend="torch"`: point
  support simple and ordinary kriging and cokriging solved in batches of equal neighbourhood size with the same
  neighbourhoods, failure rules and diagnostics; agreement with the reference within 1e-15. Universal kriging, the
  jitter policy and integrated supports are refused on the lane.
- Tests for both lanes (run locally where a CUDA device exists; reported as skipped in CI). Docs: the CUDA lane sections
  of the variogram and kriging pages, with measured timings.

## [0.05.000] - 2026-09-26

### Added

- `direct_sampling`: conditional Direct Sampling from a training image for categorical and continuous variables, with
  a seeded path, a uniform candidate permutation per node, weighted mismatch fractions, first-match acceptance
  (threshold inclusive), best-scanned fallback, masks for the TI and the domain, exact hard data with conflict and
  category checks, and per-cell candidate, score, fallback and scan records. The same definition runs as a NumPy
  reference and as a PyTorch CUDA scorer that selects the same candidates with bit-identical scores.
- `scripts/benchmark_direct_sampling.py` and its recorded result: on the tested sizes the CUDA scorer is a validated
  equal, faster in two of five cases and slower in the others.
- Docs: the Direct Sampling page with a figure of a real TI and three conditional realizations.

### Fixed

- The `cuda` extra required `torch>=2.14`, which does not exist (the latest CUDA build is 2.11.0), so
  `pip install geocond[cuda]` could not resolve; it now requires `torch>=2.9,<3`.

## [0.04.000] - 2026-09-26

### Added

- `variogram.fit_lmc`: joint fitting of a linear model of coregionalization to direct and cross variograms, each sill
  matrix and the nugget parameterized as L L^T so they are positive semidefinite by construction; variables
  standardized while fitting; deterministic starts with every start recorded. Recovers the gstat reference LMC to 1e-4.
- `probability`: an independent weighted PAVA, the bounded isotonic projection (PAVA clipped to [0, 1]) and
  `indicator_kriging`, ordinary indicator kriging per threshold on one shared neighbourhood plan, with raw and
  corrected CDFs, the correction's size, monotone interpolation between thresholds and unmodeled-tail statuses outside
  them. Agrees with scikit-learn's `IsotonicRegression` and a generic QP.
- `simulation`: `NormalScoreTransform` (weighted mid-rank plotting positions, tied values sharing one score, bounded or
  declared tails, all recorded) and `sequential_gaussian` (seeded path per realization, recorded innovations and
  conditional moments, exact hard data, native back-transform, E-type mean, quantiles and exceedance from native
  realizations). Full-neighbourhood realizations equal the dense conditional Cholesky draw to 1e-10.
- Docs: the indicator probabilities and Gaussian simulation page with two figures computed by the library, and the LMC
  fitting section of the variogram page.

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

