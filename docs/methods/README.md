# Methods

One page per implemented method: the equations, the conventions, how the implementation decides each boundary case,
what the tests establish against independent references, and the sources.

1. [Covariance models](01_covariance.md): the three families, anisotropy, the linear model of coregionalization, the
   process nugget, and the range conventions of gstat, PyKrige and GSTools.
2. [Experimental variograms and fitting](02_variograms.md): the estimators, which pairs enter which bin, pair
   sampling, and bounded deterministic fitting.
3. [Kriging and cokriging](03_kriging.md): the shared system, simple, ordinary and universal kriging, coupled
   cokriging, support integration and the nugget, measurement error, neighbourhoods, solver policy and baselines.

The contract for the methods not yet implemented is in [the API contract](../api-contract.md).
