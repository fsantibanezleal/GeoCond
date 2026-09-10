# Initial API and scientific contract

This is the implementation contract for version 0.01.000. An entry in this document is not evidence that its implementation or release gate has passed. Examples and reference receipts will accompany each completed vertical.

## Dependency and source boundary

Distribution and import are both `geocond`; repository `GeoCond`. Core dependencies are NumPy and SciPy on Python >=3.12. Torch is lazy and optional through `cuda`; installing the CPU library must not install a GPU runtime. Reference tools are optional test dependencies. No network, file acquisition, source tables, Pydantic/FastAPI, Polars/Arrow, ONNX or application UI is part of the numerical core. Serialization returns safe ordinary mappings; the consumer owns artifact files, hashes and job lifecycle.

## Geometry and support

`Survey(collar, measured_depth, azimuth, dip, *, angle_unit='degree', start_extension='error', end_extension='error')` validates arrays and evaluates minimum-curvature positions through `at(depths)`. Coordinates are local metric east/north/elevation-up; azimuth is clockwise from north and dip is negative downward. The consumer converts source bearings/units once and retains measured/assumed provenance. Antiparallel directions and unspecified extension are errors. Position inside a segment is on its circular arc, not its chord.

`Support(points, weights, *, kind, id='')` is a known sampling measure represented by quadrature, with finite (n,3) points, nonnegative normalized weights and positive total weight. Constructors `point_support`, `line_support`, `trajectory_support` and `block_support` create explicit support. Unknown-weight sampling envelopes cannot enter this type: consumers must either retain them as ineligible or explicitly choose and label a point-centre approximation outside the core. A geometric extent alone does not establish sampling weights.

`composite_intervals(from_depth, to_depth, values, boundaries, *, support_kind='continuous-interval', domains=None, source_ids=None, min_coverage=0, residual='keep')` returns coverage, valid/missing lengths, numerator, mean, domain and overlap lineage. Overlaps and unknown support are errors; NaN values represent missing observations rather than zero. Domain boundaries split output supports. Categorical compositing returns proportions through a distinct function.

## Covariance and conditioning

`CovarianceComponent(family, ranges, sill, *, rotation=None)` stores spherical/exponential/Gaussian structure, positive principal ranges, a proper orthonormal frame and a symmetric PSD sill matrix. One variable uses a (1,1) matrix. `CovarianceModel(components, *, nugget=None)` uses the same correlation/anisotropy per component across variables. Exponential correlation is exp(-3r), spherical is 1-1.5r+0.5r^3 for r<1, and Gaussian is exp(-3r^2), with transformed distance r. Range conventions are explicit in independent adapters. Process nugget and observation error are distinct.

`Observations(supports, values, *, variables=None, groups=None, ids=None, error_variance=None)` stacks actual observations, including unequal locations/support across variables. No nearest-location pseudo-pairs are created. Unknown values must be filtered explicitly before conditioning. Duplicates and degeneracies remain errors unless a declared observation-error or solver policy resolves them.

`Neighborhood(max_samples=64, min_samples=1, *, radius=inf, max_per_group=None, min_groups=1, per_variable=True, metric=None)` selects deterministic local support using support centroids and declared anisotropic distance. Group constraints require actual group identities. Neighborhood selection is not a covariance transformation.

`predict(observations, targets, model, *, method='ordinary', target_variable=0, mean=None, drift=None, neighborhood=None, solver=None, backend='numpy', cancel=None)` returns `PredictionBatch`. Methods `simple`, `ordinary` and `universal` retain their actual constraints; LMC uses the same coupled solver and one variable-indicator constraint per variable for ordinary cokriging. Explicit conveniences `simple_kriging`, `ordinary_kriging`, `universal_kriging`, `cokriging` preserve this contract. NN/IDW are separately named baselines and have no covariance variance.

`PredictionBatch` contains means, variances, valid/status arrays and one diagnostics record per target: selected observation IDs, weights, per-variable contributions/weight sums, distinct groups, residuals, covariance spectrum/condition estimate, negative-weight mass and applied regularization. Unestimated or failed targets contain NaN and an explicit reason, never fabricated zeros. Invalid shared inputs raise typed validation errors. Cancellation raises `CancelledError`; consumers must not publish a partial batch as complete.

Support-integrated covariance evaluates observation-observation, observation-target and target-target quadrature. Universal drift is averaged on each support and centered/scaled before rank checks. Linear solves never silently change UK to OK, replace full LMC by independent univariate models, or hide materially negative variance. Numerical regularization/pseudoinverse, if selected, is bounded and reported.

## Fitting and variograms

`experimental_variogram` and `experimental_cross_variogram` return bin edges, actual mean separation, pair counts, values and retained pair IDs. Direction/angular/bandwidth, robust versus classical estimator, downhole restrictions and deterministic pair sampling are explicit. Cross-variograms require genuinely aligned common supports; differing supports need an explicit restriction or another valid estimator.

`fit_variogram` and `fit_lmc` perform bounded deterministic fitting. LMC fitting parameterizes each complete sill matrix as L L-transpose; full PSD checks apply for every variable count. Fit records include objective, starts, iterations, convergence and spectra. The consumer owns train-only population selection, spatial split and model selection across fits.

## Probabilities and simulations

`indicator_kriging` accepts ordered finite thresholds and one fitted covariance per threshold, returns raw/corrected CDF and correction magnitude, and uses an independently checked bounded isotonic projection. Outside fitted thresholds returns unmodeled-tail status. No mean is inferred from unspecified tails.

`NormalScoreTransform.fit(values, weights=None)` records tie handling, weighted probabilities and bounded tail policy. `sequential_gaussian` visits a recorded seeded path, conditions on original and already simulated observations, obtains actual local SK moments, draws recorded innovations and returns native-unit realizations, paths and diagnostic masks. Original hard data remain immutable. Independent realizations may be parallel; one path is sequential.

`direct_sampling(training_image, shape, *, variable_kinds, hard_data=None, active_mask=None, ti_mask=None, max_neighbors=24, radius=inf, threshold=0, scan_fraction=1, seed=0, backend='numpy', candidate_chunk=1024, cancel=None)` supports categorical and continuous channels. TI arrays index axes in explicit (x,y,z[,variable]) order; flat export is x-fast/Fortran order. Hard coordinates/values are validated before execution. Candidate centers/offsets cannot access missing or out-of-bounds TI cells. Continuous distances use recorded scales. The first qualifying candidate in the recorded permutation wins; exhausted search uses the best actually scanned candidate and records fallback. CUDA scores chunks but preserves candidate order, path and ties. Results include candidate identity, score, fallback, path, masks and completion state; categories are copied from TI, not interpolated.

## File responsibilities and acceptance

`src/geocond/`: `validation.py`, `geometry.py`, `support.py`, `compositing.py`, `covariance.py`, `neighborhood.py`, `kriging.py`, `variogram.py`, `probability.py`, `simulation.py`, `direct_sampling.py`, `cuda.py`, `serialization.py`. Tests mirror scientific questions rather than implementation details. `tests/reference/` contains attributed authored analytic and independently executed gstat fixtures; `docs/methods/` transcribes equations, source references, failure semantics and examples alongside implementation.

Required gates include analytical arc/support/integral conservation; all covariance/constraint and full-PSD adversarial cases; gstat/PyKrige/GSTools/welleng parity with converted conventions; exact target support and quadrature convergence; variable permutation/unit scaling/zero-cross reduction; independent isotonic oracle; SGS conditional moments and hard-data honor; genuine DS scalar/CUDA candidate identity, masks, fallback and seeded structure; cancellation integrity; clean-wheel install, optional-dependency isolation, CPU/Linux/Windows and actual CUDA receipts; public source/history scan and PyPI release download verification.
