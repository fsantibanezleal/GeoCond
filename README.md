# GeoCond

GeoCond is a Python library for support-aware spatial conditioning. Its scientific core separates observation geometry and sampling support from source-specific ingestion, web services and visualization.

The first release is under active implementation. The API and acceptance contract are recorded in [docs/api-contract.md](docs/api-contract.md); this repository does not yet claim a published or validated package release.

The complete planned core covers minimum-curvature trajectories, explicit support quadrature, conservative interval compositing, directional variograms, nearest/IDW baselines, simple/ordinary/universal kriging, full positive-semidefinite LMC cokriging, multiple-indicator CDFs, sequential Gaussian simulation and genuine training-image Direct Sampling. CPU calculations use NumPy/SciPy. CUDA kernels use an optional, lazily imported PyTorch dependency and retain the CPU algorithm's conditioning order.

Source adapters, geological assumptions, grouped training splits, neural training, native SNESIM orchestration and application deployment belong to consuming products. No field datasets, credentials, personal information or infrastructure bindings are part of the library.

Licensed under Apache-2.0. Numerical methods do not by themselves establish resource classes, reserves or operational certification.
