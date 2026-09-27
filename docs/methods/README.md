# Methods

One page per implemented method: the equations, the conventions, how the implementation decides each boundary case,
what the tests establish against independent references, and the sources.

1. [Covariance models](01_covariance.md): the three families, anisotropy, the linear model of coregionalization, the
   process nugget, and the range conventions of gstat, PyKrige and GSTools.
2. [Experimental variograms and fitting](02_variograms.md): the estimators, which pairs enter which bin, pair
   sampling, bounded deterministic fitting, and joint fitting of a linear model of coregionalization.
3. [Kriging and cokriging](03_kriging.md): the shared system, simple, ordinary and universal kriging, coupled
   cokriging, support integration and the nugget, measurement error, neighbourhoods, solver policy and baselines.
4. [Indicator probabilities and Gaussian simulation](04_probability_and_simulation.md): multiple-indicator kriging
   with its order correction, the normal-score transform and its tails, and sequential Gaussian simulation.
5. [Direct Sampling](05_direct_sampling.md): the definition shared by the CPU reference and the CUDA scorer, candidate
   provenance, fallback, and a recorded benchmark of when each backend is faster.
6. [Trajectories, supports and compositing](06_geometry_and_compositing.md): the coordinate convention, minimum
   curvature with arc interpolation and explicit extensions, known sampling supports and their quadrature, and
   length-weighted and categorical compositing that keeps the sampled length.

The contract for the methods not yet implemented is in [the API contract](../api-contract.md).
