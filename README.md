# GeoCond

GeoCond is a Python library for support-aware spatial conditioning. Its scientific core separates observation geometry
and sampling support from source-specific ingestion, web services and visualization.

```bash
pip install geocond            # Python 3.12+, NumPy and SciPy only
pip install "geocond[reference]"   # plus the independent references the tests compare against
pip install "geocond[cuda]"        # plus PyTorch for the CUDA Direct Sampling scorer (a CUDA build of torch)
```

## Status

Version 0.07.000. Each capability below is implemented, tested against an independent reference, and documented; the
rest of the planned core is listed after it and is not claimed.

| Capability | Module | Checked against | Documentation |
|---|---|---|---|
| Minimum-curvature trajectories from collar and survey stations | `geometry` | analytic endpoints and welleng 0.29.1 (within 1e-9 of the path length) | [Trajectories, supports and compositing](docs/methods/06_geometry_and_compositing.md) |
| Known sampling supports and Gauss-Legendre quadrature | `support` | analytic integrals | [Trajectories, supports and compositing](docs/methods/06_geometry_and_compositing.md) |
| Conservative interval compositing, continuous and categorical | `compositing` | exact conservation | [Trajectories, supports and compositing](docs/methods/06_geometry_and_compositing.md) |
| Nested covariance: exponential, spherical and Gaussian families, geometric anisotropy, the linear model of coregionalization with full-spectrum PSD checks, process nugget | `covariance` | R/gstat 2.1-6 evaluated LMC covariances (within 2e-10) | [Covariance models](docs/methods/01_covariance.md) |
| Range conventions of gstat, PyKrige and GSTools | `conventions` | each library's own evaluated functions | [Covariance models](docs/methods/01_covariance.md) |
| Experimental direct and cross variograms: directions, angle tolerance, bandwidth, downhole pairs, classical and Cressie-Hawkins estimators, seeded pair sampling, retained pairs | `variogram` | an all-pair enumeration and GSTools 1.7.0 (identical bins) | [Variograms and fitting](docs/methods/02_variograms.md) |
| Bounded, deterministic multi-start variogram fitting, isotropic or three principal ranges | `variogram` | exact recovery of noise-free nested and anisotropic models | [Variograms and fitting](docs/methods/02_variograms.md) |
| Simple, ordinary and universal kriging and coupled simple and ordinary cokriging on support-integrated covariance, with joint error covariance, measurement error, the continuous-support nugget convention, a condition limit and per-target diagnostics | `kriging` | R/gstat 2.1-6 (univariate, coupled, block, nugget and measurement-error cases, within 2e-10), PyKrige 1.7.3 and GSTools 1.7.0 | [Kriging and cokriging](docs/methods/03_kriging.md) |
| Deterministic anisotropic neighbourhoods with per-variable and per-hole limits | `neighborhood` | stated selection rules | [Kriging and cokriging](docs/methods/03_kriging.md) |
| Nearest-neighbour and inverse-distance baselines, with no implied variance | `baselines` | direct computation | [Kriging and cokriging](docs/methods/03_kriging.md) |
| Joint LMC fitting to direct and cross variograms, every sill matrix positive semidefinite by construction (B = L L^T) | `variogram` | exact recovery of the gstat reference LMC and a random three-variable LMC | [Variograms and fitting](docs/methods/02_variograms.md) |
| Multiple-indicator kriging with the bounded isotonic (PAVA) correction and unmodeled-tail statuses | `probability` | scikit-learn `IsotonicRegression` and a generic QP | [Indicator probabilities and Gaussian simulation](docs/methods/04_probability_and_simulation.md) |
| Weighted normal-score transform with recorded tie and tail policies; sequential Gaussian simulation with seeded paths, recorded innovations and exact hard data | `simulation` | the dense Gaussian conditional (exact, to 1e-10) and its moments | [Indicator probabilities and Gaussian simulation](docs/methods/04_probability_and_simulation.md) |
| Float64 CUDA lanes (PyTorch): every variogram pair tiled on the device; point-support simple and ordinary kriging and cokriging solved in batches | `cuda`, `variogram`, `kriging` | the NumPy reference: identical counts, bins within 1e-12, predictions within 1e-15 | [Variograms](docs/methods/02_variograms.md), [Kriging](docs/methods/03_kriging.md) |
| Direct Sampling from a training image, categorical and continuous, with per-cell candidate provenance, fallback and completion records; a PyTorch CUDA scorer that selects the same candidates with bit-identical scores | `direct_sampling` | a plain-Python enumeration of the definition; the TI's conditional frequencies; CPU against CUDA, locally | [Direct Sampling](docs/methods/05_direct_sampling.md) |

Planned and not yet claimed: training-image Direct Sampling; the optional PyTorch CUDA kernels that retain the CPU algorithm's
conditioning order. The contract for all of them is in [docs/api-contract.md](docs/api-contract.md).

## Scope

Source adapters, geological assumptions, grouped training splits, neural training, native SNESIM orchestration and
application deployment belong to consuming products. No field datasets, credentials, personal information or
infrastructure bindings are part of the library.

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[test,reference,dev]"   # .venv/bin/python elsewhere
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/figures/methods_figures.py          # rebuild the method figures
```

The reference fixture `tests/reference/gstat_fixed_covariance.json` holds authored inputs and the outputs R/gstat
computed for them in an isolated container; it carries its attribution and no gstat code.

Licensed under Apache-2.0. Numerical methods do not by themselves establish resource classes, reserves or operational
certification.
