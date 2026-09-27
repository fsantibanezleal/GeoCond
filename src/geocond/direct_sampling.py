"""Direct Sampling: conditional multiple-point simulation that copies values from a training image.

Direct Sampling (Mariethoz, Renard and Straubhaar 2010) visits the simulation nodes on a random path. At each node it
takes the pattern of informed neighbours (hard data and nodes already simulated), scans the training image (TI) for a
location whose neighbourhood matches that pattern closely enough, and copies the value at that location. It stores no
pattern tree and handles categorical and continuous variables alike.

The definition shared by the NumPy reference and the PyTorch (CUDA) scorer, so both select the same candidate:

1. The path is a seeded permutation of the active cells without hard data (``default_rng([seed, 0])``).
2. A node's neighbours are the ``max_neighbors`` nearest informed cells within ``radius`` (Euclidean, in cells), ties
   broken by flat index; their offsets and values form the data event.
3. The candidates are the valid TI centres, visited in a fresh uniform permutation per node
   (``default_rng([seed, 1, node])``), so the first qualifying candidate is uniform among qualifying candidates and the
   copied values follow the TI's conditional frequencies. A candidate whose pattern would leave the TI or touch a missing
   TI cell is skipped: the TI is not wrapped.
4. The mismatch is a weighted fraction: for a categorical variable sum w_i [TI(c + d_i) != v_i]; for a continuous one
   sqrt(sum w_i ((TI(c + d_i) - v_i) x (1/r))^2) with r the TI's range (a training-only scale); the variables' scores
   are summed and multiplied by 1/(number of variables). The weights are |d_i|^-distance_power, normalized. Every
   operation is an IEEE float64 add, multiply or square root in the same order on both backends (sums over
   neighbours one addition at a time, and multiplications by precomputed reciprocals rather than divisions, which
   PyTorch would otherwise turn into reciprocal multiplications on its own), so the scores agree bit for bit.
5. The first candidate in the node's order with mismatch <= threshold wins. When ``scan_fraction`` of the candidates
   has been examined without one, the best candidate actually scanned (earliest on ties) is used and the fallback is
   recorded. A node with no valid candidate at all is marked failed and left unwritten.
6. The centre's values are copied: categories are TI codes, never interpolated. Hard data are never altered.
7. With ``zones``, a node's candidates are only the valid TI centres of the node's zone, and step 3's permutation and
   the scan fraction apply to that zone's centres. This is the zonal treatment of nonstationarity (Mariethoz, Renard
   and Straubhaar 2010, section 6: "scanning a different part of a TI ... for each simulated zone"): a layered prior,
   for instance, keeps its order when each depth layer of the grid scans the same layer of the TI. With every zone
   equal the result is identical to the unzoned engine.

Scanning candidates in chunks on a GPU preserves these semantics: a chunk is scored in parallel, but the earliest
qualifying candidate in the node's order is taken, never the best of the chunk.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import ceil, inf

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import ValidationError, cancel_if_requested, integer, positive

KINDS = ("categorical", "continuous")


@dataclass(frozen=True)
class DirectSamplingResult:
    """One realization with the provenance of every simulated cell. Arrays are indexed (x, y, z[, variable]); flat
    indices (``path``, ``candidate``) are x-fast (Fortran order) into the grid and the TI respectively."""

    realization: NDArray[np.float64]
    candidate: NDArray[np.int64]
    score: NDArray[np.float64]
    fallback: NDArray[np.bool_]
    scanned: NDArray[np.int64]
    path: NDArray[np.int64]
    hard: NDArray[np.bool_]
    active: NDArray[np.bool_]
    failed: NDArray[np.bool_]
    complete: bool
    seed: int
    backend: str
    parameters: dict = field(default_factory=dict)


def _prepare(training_image, shape, variable_kinds, hard_data, active_mask, ti_mask):
    ti = np.asarray(training_image, dtype=np.float64)
    if ti.ndim == 3:
        ti = ti[..., None]
    if ti.ndim != 4:
        raise ValidationError("the training image must be (nx, ny, nz) or (nx, ny, nz, variables)")
    kinds = tuple(variable_kinds)
    if len(kinds) != ti.shape[3] or any(k not in KINDS for k in kinds):
        raise ValidationError(f"one kind from {KINDS} per training-image variable is required")
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValidationError("shape must be three positive grid dimensions")
    ti_valid = np.all(np.isfinite(ti), axis=3)
    if ti_mask is not None:
        mask = np.asarray(ti_mask, dtype=bool)
        if mask.shape != ti.shape[:3]:
            raise ValidationError("ti_mask must match the training image's grid")
        ti_valid &= mask
    if not ti_valid.any():
        raise ValidationError("the training image has no valid cell")
    active = np.ones(shape, bool) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != shape:
        raise ValidationError("active_mask must match the simulation grid")
    values = ti[ti_valid]
    categories = [np.unique(values[:, v]) if k == "categorical" else None for v, k in enumerate(kinds)]
    scale = np.array([
        1.0 if k == "categorical" or np.ptp(values[:, v]) == 0 else float(np.ptp(values[:, v]))
        for v, k in enumerate(kinds)
    ])
    sim = np.full(shape + (len(kinds),), np.nan)
    hard = np.zeros(shape, bool)
    if hard_data is not None:
        idx, hv = hard_data
        idx = np.asarray(idx)
        hv = np.asarray(hv, dtype=np.float64)
        if hv.ndim == 1:
            hv = hv[:, None]
        if idx.ndim != 2 or idx.shape[1] != 3 or idx.dtype.kind not in "iu" or len(idx) != len(hv):
            raise ValidationError("hard data are (m, 3) integer cell indices and m values (per variable)")
        if hv.shape[1] != len(kinds) or not np.isfinite(hv).all():
            raise ValidationError("hard values must be finite, one per variable")
        if np.any(idx < 0) or np.any(idx >= np.array(shape)):
            raise ValidationError("hard data must lie inside the simulation grid")
        for v, cats in enumerate(categories):
            if cats is not None and not np.isin(hv[:, v], cats).all():
                raise ValidationError(f"hard data of variable {v} hold a category the training image does not have")
        for (i, j, k), val in zip(idx, hv, strict=True):
            if hard[i, j, k] and not np.array_equal(sim[i, j, k], val):
                raise ValidationError(f"conflicting hard data in cell ({i}, {j}, {k})")
            hard[i, j, k] = True
            sim[i, j, k] = val
    return ti, kinds, shape, ti_valid, active, scale, sim, hard, categories


class _NumpyScorer:
    def __init__(self, ti, ti_valid, kinds, scale):
        self.ti, self.valid, self.kinds = ti, ti_valid, kinds
        self.inverse = [1.0 / float(s) for s in scale]
        self.share = 1.0 / len(kinds)
        self.dims = np.array(ti.shape[:3])

    def first(self, centers, offsets, values, weights, threshold):
        """Scores of a chunk of candidate centres, then the first qualifying position and the best valid one."""
        pos = centers[:, None, :] + offsets[None, :, :]
        inside = np.all((pos >= 0) & (pos < self.dims), axis=2)
        ok = inside.all(axis=1)
        safe = np.where(inside[..., None], pos, 0)
        cell_valid = self.valid[safe[..., 0], safe[..., 1], safe[..., 2]]
        ok &= cell_valid.all(axis=1)
        tv = self.ti[safe[..., 0], safe[..., 1], safe[..., 2]]  # (C, n, variables)
        score = np.zeros(len(centers))
        for v, kind in enumerate(self.kinds):
            acc = np.zeros(len(centers))
            for i in range(offsets.shape[0]):
                if kind == "categorical":
                    acc = acc + weights[i] * (tv[:, i, v] != values[i, v])
                else:
                    diff = (tv[:, i, v] - values[i, v]) * self.inverse[v]
                    acc = acc + weights[i] * (diff * diff)
            score = score + (acc if kind == "categorical" else np.sqrt(acc))
        score = score * self.share
        qualifying = ok & (score <= threshold)
        first = int(np.argmax(qualifying)) if qualifying.any() else -1
        best = -1
        if ok.any():
            masked = np.where(ok, score, np.inf)
            best = int(np.argmin(masked))
        return first, best, score


class _TorchScorer:
    def __init__(self, ti, ti_valid, kinds, scale, device):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.ti = torch.as_tensor(ti, dtype=torch.float64, device=self.device)
        self.valid = torch.as_tensor(ti_valid, dtype=torch.bool, device=self.device)
        self.kinds = kinds
        self.inverse = [1.0 / float(s) for s in scale]
        self.share = 1.0 / len(kinds)
        self.dims = torch.as_tensor(ti.shape[:3], dtype=torch.int64, device=self.device)

    def first(self, centers, offsets, values, weights, threshold):
        torch = self.torch
        c = torch.as_tensor(centers, dtype=torch.int64, device=self.device)
        d = torch.as_tensor(offsets, dtype=torch.int64, device=self.device)
        vals = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        w = [float(x) for x in weights]
        pos = c[:, None, :] + d[None, :, :]
        inside = ((pos >= 0) & (pos < self.dims)).all(dim=2)
        ok = inside.all(dim=1)
        safe = torch.where(inside[..., None], pos, torch.zeros_like(pos))
        ok = ok & self.valid[safe[..., 0], safe[..., 1], safe[..., 2]].all(dim=1)
        tv = self.ti[safe[..., 0], safe[..., 1], safe[..., 2]]
        score = torch.zeros(len(centers), dtype=torch.float64, device=self.device)
        for v, kind in enumerate(self.kinds):
            acc = torch.zeros(len(centers), dtype=torch.float64, device=self.device)
            for i in range(offsets.shape[0]):
                if kind == "categorical":
                    acc = acc + w[i] * (tv[:, i, v] != vals[i, v]).to(torch.float64)
                else:
                    diff = (tv[:, i, v] - vals[i, v]) * self.inverse[v]
                    acc = acc + w[i] * (diff * diff)
            score = score + (acc if kind == "categorical" else torch.sqrt(acc))
        score = score * self.share
        qualifying = ok & (score <= threshold)
        first = int(torch.argmax(qualifying.to(torch.int8)).item()) if bool(qualifying.any()) else -1
        best = -1
        if bool(ok.any()):
            best = int(torch.argmin(torch.where(ok, score, torch.full_like(score, inf))).item())
        return first, best, score.cpu().numpy()


def direct_sampling(
    training_image: ArrayLike,
    shape: Sequence[int],
    *,
    variable_kinds: Sequence[str],
    hard_data: tuple[ArrayLike, ArrayLike] | None = None,
    active_mask: ArrayLike | None = None,
    ti_mask: ArrayLike | None = None,
    max_neighbors: int = 24,
    radius: float = inf,
    threshold: float = 0.0,
    scan_fraction: float = 1.0,
    seed: int = 0,
    backend: str = "numpy",
    device: str = "cuda",
    candidate_chunk: int | None = None,
    distance_power: float = 0.0,
    zones: tuple[ArrayLike, ArrayLike] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> DirectSamplingResult:
    """One Direct Sampling realization on a grid of ``shape`` cells from ``training_image`` (see the module text for
    the exact definition). ``backend='torch'`` scores candidate chunks with PyTorch on ``device`` and selects the same
    candidates as the NumPy reference. ``candidate_chunk`` defaults to 1,024 candidates on NumPy and 65,536 on PyTorch,
    the sizes at which each was fastest in the recorded benchmark; the chunk never changes the result. ``zones`` is
    ``(ti_zones, grid_zones)``, one integer per TI cell and per grid cell (step 7 of the module text)."""
    ti, kinds, shape, ti_valid, active, scale, sim, hard, _ = _prepare(
        training_image, shape, variable_kinds, hard_data, active_mask, ti_mask
    )
    ti_zones, grid_zones = _zones(zones, ti_valid.shape, shape)
    max_neighbors = integer(max_neighbors, "max_neighbors")
    if not (radius == inf or positive(radius, "radius")):
        raise ValidationError("radius must be positive or infinite")
    threshold = positive(threshold, "threshold", zero=True)
    if not 0 < scan_fraction <= 1:
        raise ValidationError("scan_fraction must lie in (0, 1]")
    if candidate_chunk is None:
        candidate_chunk = 65536 if backend == "torch" else 1024
    candidate_chunk = integer(candidate_chunk, "candidate_chunk")
    distance_power = positive(distance_power, "distance_power", zero=True)
    if backend == "numpy":
        scorer = _NumpyScorer(ti, ti_valid, kinds, scale)
    elif backend == "torch":
        scorer = _TorchScorer(ti, ti_valid, kinds, scale, device)
    else:
        raise ValidationError("backend must be 'numpy' or 'torch'")

    centers = np.argwhere(ti_valid)  # C order over (x, y, z); ids below are Fortran-order flat indices
    center_ids = np.ravel_multi_index(tuple(centers.T), ti_valid.shape, order="F")
    n_centers = len(centers)
    # The candidates of each zone, as positions into ``centers`` in its order; one zone of every centre when unzoned.
    zone_of_center = ti_zones[tuple(centers.T)]
    by_zone = {int(z): np.flatnonzero(zone_of_center == z) for z in np.unique(zone_of_center)}
    empty = np.zeros(0, np.int64)

    nodes = np.flatnonzero((active & ~hard).ravel(order="F"))
    path = nodes[np.random.default_rng([seed, 0]).permutation(len(nodes))]
    hard_xyz = np.argwhere(hard)
    capacity = len(hard_xyz) + len(path)
    all_xyz = np.zeros((capacity, 3), np.int64)
    all_val = np.zeros((capacity, len(kinds)))
    all_id = np.zeros(capacity, np.int64)
    count = len(hard_xyz)
    all_xyz[:count] = hard_xyz
    all_val[:count] = sim[hard]
    all_id[:count] = np.ravel_multi_index(tuple(hard_xyz.T), shape, order="F") if count else []

    candidate = np.full(shape, -1, np.int64)
    score_out = np.full(shape, np.nan)
    fallback = np.zeros(shape, bool)
    scanned = np.zeros(shape, np.int64)
    failed = np.zeros(shape, bool)

    for node in path:
        cancel_if_requested(cancel)
        xyz = np.array(np.unravel_index(node, shape, order="F"))
        inf_xyz, inf_val, inf_id = all_xyz[:count], all_val[:count], all_id[:count]
        if count:
            dist = np.sqrt(((inf_xyz - xyz) ** 2).sum(axis=1))
            order = np.lexsort((inf_id, dist))
            order = order[dist[order] <= radius][:max_neighbors]
        else:
            order = np.zeros(0, np.int64)
        offsets = inf_xyz[order] - xyz
        values = inf_val[order]
        if len(order):
            raw = np.linalg.norm(offsets, axis=1) ** (-distance_power) if distance_power else np.ones(len(order))
            weights = raw / raw.sum()
        else:
            weights = np.zeros(0)
        pool = by_zone.get(int(grid_zones[tuple(xyz)]), empty)
        permutation = pool[np.random.default_rng([seed, 1, int(node)]).permutation(len(pool))]
        max_scan = max(1, ceil(scan_fraction * len(pool))) if len(pool) else 0
        chosen = -1
        best, best_score = -1, inf
        examined = 0
        for start in range(0, max_scan, candidate_chunk):
            stop = min(max_scan, start + candidate_chunk)
            chunk = permutation[start:stop]
            first, best_in_chunk, scores = scorer.first(centers[chunk], offsets, values, weights, threshold)
            if first >= 0:
                chosen = int(chunk[first])
                examined = start + first + 1
                chosen_score = float(scores[first])
                break
            if best_in_chunk >= 0 and scores[best_in_chunk] < best_score:
                best, best_score = int(chunk[best_in_chunk]), float(scores[best_in_chunk])
            examined = stop
        i, j, k = xyz
        scanned[i, j, k] = examined
        if chosen < 0:
            if best < 0:
                failed[i, j, k] = True
                continue
            chosen, chosen_score = best, best_score
            fallback[i, j, k] = True
        c = centers[chosen]
        sim[i, j, k] = ti[c[0], c[1], c[2]]
        candidate[i, j, k] = center_ids[chosen]
        score_out[i, j, k] = chosen_score
        all_xyz[count] = xyz
        all_val[count] = sim[i, j, k]
        all_id[count] = node
        count += 1

    return DirectSamplingResult(
        sim, candidate, score_out, fallback, scanned, path, hard, active, failed, not failed.any(), int(seed), backend,
        {"max_neighbors": max_neighbors, "radius": radius, "threshold": threshold, "scan_fraction": scan_fraction,
         "candidate_chunk": candidate_chunk, "distance_power": distance_power, "variable_kinds": list(kinds),
         "continuous_scales": scale.tolist(), "valid_centers": int(n_centers),
         "zones": None if zones is None else {str(z): len(c) for z, c in by_zone.items()}},
    )


def _zones(zones, ti_shape, grid_shape):
    """Integer zone arrays for the TI and the grid; one shared zone when ``zones`` is None."""
    if zones is None:
        return np.zeros(ti_shape, np.int64), np.zeros(grid_shape, np.int64)
    try:
        ti_zones, grid_zones = (np.asarray(z) for z in zones)
    except (TypeError, ValueError) as error:
        raise ValidationError("zones must be (ti_zones, grid_zones)") from error
    for name, z, expected in (("ti_zones", ti_zones, ti_shape), ("grid_zones", grid_zones, grid_shape)):
        if z.shape != tuple(expected):
            raise ValidationError(f"{name} must have the shape {tuple(expected)}")
        if z.dtype.kind not in "iu":
            raise ValidationError(f"{name} must hold integers")
    return ti_zones.astype(np.int64), grid_zones.astype(np.int64)
